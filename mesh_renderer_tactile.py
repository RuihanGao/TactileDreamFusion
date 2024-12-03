import os
import math
import cv2
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
from mesh_tactile import Mesh, safe_normalize
from utils import dot, create_uv_coords
from neural_style_field import NeuralStyleField, VanillaMLP, NetworkWithInputEncoding
from dataclasses import dataclass, field
import time
import pdb

### ChangeLog ###
# 20240315: add a tactile texture map to the renderer 
# 20240411: add "part label" map to the renderer. The label map is used to map different parts of the object to different tactile textures.
# 20240412: instead of rendering "part label", we render SAM features. 
# 20240421: add options to use field representations for albedo and tactile texture

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

def trunc_rev_sigmoid(x, eps=1e-6):
    # change range from [0, 1] to [-inf, inf]
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

def make_divisible(x, m=8):
    return int(math.ceil(x / m) * m)


def uv_padding(image, hole_mask, xatlas_pack_options=field(default_factory=dict)):
    # ref: https://github.com/threestudio-project/threestudio/blob/cd462fb0b73a89b6be17160f7802925fe6cf34cd/threestudio/models/exporters/mesh_exporter.py#L93
   
    # uv_padding_size = xatlas_pack_options.get("padding", 2) -> inpaintRadius parameter in inpaint function
    """
    Input: 
        image: torch tensor of shape [H, W, 3], range [0, 1]
    
    Output:
        inpaint_image: torch tensor of shape [H, W, 3], range [0, 1]
    """
    
    inpaint_image = (
        cv2.inpaint(
            (image.detach().cpu().numpy() * 255).astype(np.uint8),
            (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
            2,
            cv2.INPAINT_TELEA,
        )
        / 255.0
    )
    return torch.from_numpy(inpaint_image).to(image)



class Renderer(nn.Module):
    def __init__(self, opt):
        
        super().__init__()

        self.opt = opt

        self.mesh = Mesh.load(self.opt.mesh, resize=False, opt=self.opt)
        print(f"Loaded mesh from {self.opt.mesh} N_v {len(self.mesh.v)} N_f {len(self.mesh.f)}") # N_v 19882 N_f 35972

        self.glctx = dr.RasterizeGLContext()
        # self.glctx = dr.RasterizeCudaContext()
        
        # extract trainable parameters
        self.v_offsets = nn.Parameter(torch.zeros_like(self.mesh.v))

        self.device = self.v_offsets.device
        
        # Try initializing the mesh albedo map with grayscale color during refinement
        # print(f"check mesh.albedo shape {self.mesh.albedo.shape}, range [{self.mesh.albedo.min()},{self.mesh.albedo.max()}], dtype {self.mesh.albedo.dtype} device {self.mesh.albedo.device}")

        # check mesh.albedo shape torch.Size([1024, 1024, 3]), range [0.0,0.49803921580314636], dtype torch.float32 device cuda:0
        # check raw_albedo parameter for mesh renderer, shape torch.Size([1024, 1024, 3]), range [-13.815509796142578,-0.007843117229640484], dtype torch.float32 device cuda:0

        # create a field to represent albedo and tactile texture
        self.texture_field = NeuralStyleField(
            width=64, 
            depth=0,
            colordepth=1, 
            normdepth=1,
            labeldepth=1,
            input_dim=3, # (x, y, z)
            color_output_dim=3,
            normal_output_dim=3,
            label_output_dim=2, # NOTE: label_output_dim should be the same number of labels/parts
            num_frequencies=self.opt.num_frequencies,
            min_freq = 0.0,
            max_freq = self.opt.max_freq,
            use_tcnn_encoding = self.opt.use_tcnn_encoding,
            print_model = False,
        ).to(self.device)



        self.query_texture() # set up self.raw_albedo and self.tactile_normal


        # Set the target albdeo and tactile normal map for 1. initialization and 2. regularization during refinement
        # Here use uv map as the target representations
        self.target_albedo = self.mesh.albedo.clone()
        
        if self.opt.load_tactile:
            self.target_perturb_normal = self.mesh.tactile_normal.clone().detach().to(self.device).contiguous()
            # resize it to texture map size
            if self.target_perturb_normal.shape[0] != self.mesh.texture_map_size:
                # reshape target_perturb_normal to match tactile_normal
                target_perturb_normal = self.target_perturb_normal.permute(2, 0, 1).unsqueeze(0) # shape [1, 3, H, W]
                self.target_perturb_normal = F.interpolate(target_perturb_normal, (self.mesh.texture_map_size, self.mesh.texture_map_size), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0).contiguous() # shape [texture_size, texture_size, 3]

            if self.opt.num_part_label > 0:
                assert self.opt.num_part_label == 2, "Only support two-parts for now"
                self.target_perturb_normal2 = self.mesh.tactile_normal2.clone().detach().to(self.device).contiguous()
                # resize it to texture map size
                if self.target_perturb_normal2.shape[0] != self.mesh.texture_map_size:
                    # reshape target_perturb_normal to match tactile_normal
                    target_perturb_normal2 = self.target_perturb_normal2.permute(2, 0, 1).unsqueeze(0)
                    self.target_perturb_normal2 = F.interpolate(target_perturb_normal2, (self.mesh.texture_map_size, self.mesh.texture_map_size), mode="bilinear", align_corners=False).squeeze(0).permute(1,2,0).contiguous() # shape [texture_size, texture_size, 3]
        else:
            # we don't have tactile texture, create a dummy target_perturb_normal
            self.target_perturb_normal = torch.tensor([0, 0, 1], dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)
            # TODO: check shape and see whether we need it at all
        

        

    def query_texture(self, use_uv_padding=False):
        # Ref: threestudio mesh_exporter.py https://github.com/threestudio-project/threestudio/blob/cd462fb0b73a89b6be17160f7802925fe6cf34cd/threestudio/models/exporters/mesh_exporter.py#L72
        # v_tex - mesh.vt
        # t_tex_indices - mesh.ft

        print(f"In query_texture for 3dfield")
        # print("check self.mesh.vt shape {self.mesh.vt.shape} range [{self.mesh.vt.min()},{self.mesh.vt.max()}], self.meth.ft shape {self.mesh.ft.shape} range [{self.mesh.ft.min()},{self.mesh.ft.max()}]") # self.mesh.vt shape torch.Size([177946, 2]) range [0.0,0.999605655670166], self.meth.ft shape torch.Size([237140, 3]) range [0, 177945]

        # convert range to [-1, 1]
        uv_clip = self.mesh.vt * 2 - 1 # shape [N_v, 2]
        # pad to four component coordinate. 
        uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., 0:1]), torch.ones_like(uv_clip[..., 0:1])), dim=-1) # shape [N_v, 4], float32
        # rasterize
        rast, _ = dr.rasterize(self.glctx, uv_clip4.unsqueeze(0), self.mesh.ft, (self.mesh.texture_map_size, self.mesh.texture_map_size))
        hole_mask = ~((rast[0, ..., 3:] > 0)) # True for holes
        fg_mask = rast[0, ..., 3:] > 0 # True for foreground, shape [texture_size, texture_size, 1]
        # Interpolate the world space position
        v_pos, _ = dr.interpolate(self.mesh.v.unsqueeze(0), rast, self.mesh.f)
        v_pos = v_pos.squeeze(0) # shape [texture_map_size, texture_map_size, 3]
        # Sample out textures from the texture field
        rendered_texture_map = self.texture_field(v_pos.view(-1, 3)).view(self.mesh.texture_map_size, self.mesh.texture_map_size, -1) # shape [texture_size, texture_size, 8], range [0,1], dtype float32
        self.raw_albedo = rendered_texture_map[..., :3] # shape [texture_size, texture_size, 3]
        if self.opt.load_tactile:
            self.tactile_normal = rendered_texture_map[..., 3:6]
        self.label_map = torch.zeros_like(self.raw_albedo)
        rendered_labels = rendered_texture_map[..., 6:] # shape [texture_size, texture_size, 2]
        # based on the last channel of rendered_labels, create a mask for partA and partB
        predicted_labels = torch.argmax(rendered_labels, dim=-1).unsqueeze(-1) # [texture_size, texture_size, 1]
        print(f"check predicted_labels shape {predicted_labels.shape}, fg_mask shape {fg_mask.shape}")
        partA_mask_rendered = (predicted_labels == 0).float() * fg_mask # [texture_size, texture_size, 1], float
        partB_mask_rendered = (predicted_labels == 1).float() * fg_mask #
        # concate partA_mask and partB_mask to form the rendered seg_masks
        predicted_labels = torch.cat([partA_mask_rendered, partB_mask_rendered], dim=-1).float() # shape [512, 512, 2]
        self.label_map[..., :2]  = predicted_labels
        
        if use_uv_padding:
            # Perform UV padding on texutre maps to avoid seams, only useful for final mesh export
            # It takes about 2 min to save a uv map of size 1024x1024
            print(f"Perform UV padding on texutre maps to avoid seams, may take a while ...")
            start_time = time.time()
            self.raw_albedo = uv_padding(self.raw_albedo, hole_mask)
            # need to convert 2 channel label map to 3 channel before padding
            self.label_map = uv_padding(self.label_map, hole_mask)
            if self.opt.load_tactile:
                # convert tactile_normal to [0, 1] range before padding
                self.tactile_normal = (self.tactile_normal + 1) / 2
                self.tactile_normal = uv_padding(self.tactile_normal, hole_mask)
                # convert back to [-1, 1] range
                self.tactile_normal = self.tactile_normal * 2 - 1
            print(f"Finish UV padding in {time.time() - start_time} seconds")
            # shape [1024, 1024, 3], range [-1, 1]



    def get_params(self):
        params = []
        params.append({'params': self.texture_field.parameters(), 'lr': self.opt.texture_lr})
        return params

    @torch.no_grad()
    def export_mesh(self, save_path):
        # save v_offsets for debugging
        v_offsets = self.v_offsets.detach().cpu().numpy()
        output_path = save_path.replace('.obj', '_v_offsets.npy')
        print(f"Saving v_offsets to {output_path}")
        np.save(output_path, v_offsets)

        self.mesh.v = (self.mesh.v + self.v_offsets).detach()

        # export 3d texture field to uv map for exporting
        self.query_texture(use_uv_padding=True)
        self.mesh.albedo = self.raw_albedo.detach()
        self.mesh.label_map = self.label_map.detach()
        if self.opt.load_tactile:
            self.mesh.tactile_normal = self.tactile_normal.detach()

        self.mesh.write(save_path)

    
    def render(self, pose, proj, h0, w0, ssaa=1, bg_color=1, texture_filter='linear-mipmap-linear', shading='diffuse'):
        """
        Args:
            pose: camera pose in world space, shape [4, 4]
        """
        
        # do super-sampling
        if ssaa != 1:
            h = make_divisible(h0 * ssaa, 8)
            w = make_divisible(w0 * ssaa, 8)
        else:
            h, w = h0, w0
        
        results = {}

        # get v
        v = self.mesh.v

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

        #############################
        # Rasterize
        #############################
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T
        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (h, w))
        mask = (rast[0, ..., 3:] > 0).float().unsqueeze(0) # [1, H, W, 1], range [0, 1]
        mask_aa = dr.antialias(mask, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 1], range [0, 1]
        selector = mask[..., 0] # foreground mask, shape [1, H, W], range [0, 1]

        #############################
        # Interpolate attributes
        #############################
        alpha = (rast[0, ..., 3:] > 0).float()
        depth, _ = dr.interpolate(-v_cam[..., [2]], rast, self.mesh.f) # [1, H, W, 1]
        depth = depth.squeeze(0) # [H, W, 1]
        # interpolate texture coordinates
        texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
        # interpolate vertex position
        v_pos, _ = dr.interpolate(v.unsqueeze(0), rast, self.mesh.f) # [1, H, W, 3], range [-1, 1]
        H, W = v_pos.shape[1:3]

        # For 2d/3d field representation, but self.raw_albedo and self.tactile_normal are not used in render() function; they are only used in exporting mesh (for now)
        # self.query_texture() # udpate self.raw_albedo and self.tactile_normal, 
        
        # first interpoalte the mesh vertices pos (done above), then run forward pass of the texture field
        rendered_texture = self.texture_field(v_pos.view(-1, 3)).view(-1, v_pos.shape[1], v_pos.shape[2], 8) # shape: [1, H, W, 8]
        albedo = rendered_texture[..., :3].contiguous() # [1, H, W, 3]
        if self.opt.load_tactile:
            perturb_normal = rendered_texture[..., 3:6].contiguous()
        label_map = rendered_texture[..., 6:].contiguous() # [1, H, W, 1], torch.float32

    
        # Render target_albedo and target_tactile_normal for regularization of rendered views. 
        target_albedo = dr.texture(self.target_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode=texture_filter) # [1, H, W, 3]
        if self.opt.load_tactile:
            target_perturb_normal = dr.texture(self.target_perturb_normal.unsqueeze(0), texc, uv_da=texc_db, filter_mode=texture_filter).to(torch.float32)
            if self.opt.num_part_label > 0:
                target_perturb_normal2 = dr.texture(self.target_perturb_normal2.unsqueeze(0), texc, uv_da=texc_db, filter_mode=texture_filter).to(torch.float32)
        
        else:
            # create a map of [0, 0, 1]
            perturb_normal = torch.tensor([0, 0, 1], dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(rast.device)
            # repeat to shape [1, H, W, 3]
            perturb_normal = perturb_normal.repeat(1, H, W, 1) 
            target_perturb_normal = perturb_normal.clone() # create a dummy target_perturb_normal
        target_perturb_normal = safe_normalize(target_perturb_normal) # shape [1, H, W, 3]
        
        # print(f"check target map in mesh renderer. target_albedo shape {target_albedo.shape}, range [{target_albedo.min()}, {target_albedo.max()}], target_perturb_normal shape {target_perturb_normal.shape} range [{target_perturb_normal.min()}, {target_perturb_normal.max()}]") # [1, H, W, 3], range [0,1]; [1, H, W, 3], range [-1, 1]     

        # get vn and render normal
        vn = self.mesh.vn
        
        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
        normal = safe_normalize(normal[0]) # shape  [H, W, 3]
        H, W, _ = normal.shape

        # compute tangent
        tangent, _ = dr.interpolate(self.mesh.v_tangent.unsqueeze(0).contiguous(), rast, self.mesh.fn)
        tangent = safe_normalize(tangent[0]) # shape [H, W, 3]

        bitangent = safe_normalize(torch.cross(tangent, normal, dim=-1)) # shape [H, W, 3]
        perturb_normal = safe_normalize(perturb_normal) # shape [H, W, 3]
        shading_normal = tangent * perturb_normal[..., [0]] - bitangent * perturb_normal[..., [1]] + normal * perturb_normal[..., [2]] # shape [1, H, W, 3]
        shading_normal = safe_normalize(shading_normal.view(-1, 3))# shape [H*W, 3]

        # use shading_normal to create view-dependent shading
        # 2024-04-19: set up light source for rendering
        # TODO: currently we render one image at a time, i.e. B = 1 and no dimension for batch size. Check whether it is sufficient
        if self.opt.light_sample_strategy == "camdir":
            # create point light source at the camera position and look at the object
            light_positions = pose[:3, 3].unsqueeze(0).expand(h*w, -1) # pose shape [4, 4], shape torch.Size([36864, 3])
            # print(f"In function render, opt.light_sample_strategy is {self.opt.light_sample_strategy}, light_positions shape {light_positions.shape}")
        
        # Modify light_sample_strategy from threestudio/data/uncond.py. In threestudio, light sampling has shape [B, 3], while here we have shape [3] then expand to [H*W, 3]
        elif self.opt.light_sample_strategy == "dreamfusion":
            # TODO: make the params light_position_perturb and light_distance_range configurable in the opt
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_position_perturb = 1.0
            light_direction = F.normalize(pose[:3, 3] + torch.randn(3, dtype=torch.float32, device=pose.device) * light_position_perturb, dim=-1) # torch.Size([3])
            # get light position by scaling light direction by light distance
            light_distance_range = (3, 5)
            light_distance = (torch.rand(1, dtype=torch.float32, device=pose.device)* (light_distance_range[1] - light_distance_range[0])+ light_distance_range[0])
            light_position = light_direction * light_distance # shape torch.Size([3])
            light_positions = light_position.unsqueeze(0).expand(h*w, -1)
    
        elif self.opt.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(pose[:3, 3], dim=-1) # shape torch.Size([3])
            local_x = F.normalize(
                torch.stack(
                    [local_z[1], -local_z[0], torch.zeros_like(local_z[0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1) # shape torch.Size([3, 3])
            light_azimuth = (
                torch.rand(1, dtype=torch.float32, device=pose.device) * math.pi * 2 - math.pi
            )  # torch.Size([1]) in range [-pi, pi]
            light_elevation = (
                torch.rand(1, dtype=torch.float32, device=pose.device) * math.pi / 3 + math.pi / 6
            )  # torch.Size([1]) in range [pi/6, pi/2]
            light_distance_range = (3, 5)
            light_distance = (torch.rand(1, dtype=torch.float32, device=pose.device)* (light_distance_range[1] - light_distance_range[0])+ light_distance_range[0])
            light_positions_local = torch.stack(
                [
                    light_distance
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distance
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distance * torch.sin(light_elevation),
                ],
                dim=-1,
            ) # shape torch.Size([1, 3])
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0] # shape [1, 3, 1] -> [1, 3]
            light_positions = light_positions.expand(h*w, -1) # shape [H*W, 3]
        
        else:
            raise ValueError(f"Unknown light_sample_strategy {self.opt.light_sample_strategy}")


        light_directions = safe_normalize(light_positions - v_pos[..., :3].view(-1, 3)) # [H*W, 3]
        diffuse_light_color = self.opt.diffuse_light_color * torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=light_directions.device) # shape [3]
        ambient_light_color = self.opt.ambient_light_color * torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=light_directions.device)

        # Ref: threestudio/models/materials/diffuse_with_point_light_material.py
        diffuse_light = dot(shading_normal.detach(), light_directions).clamp(min=0.0) * diffuse_light_color # shape [H*W, 1] * shape [3] = shape [H*W, 3]
        textureless_color = (diffuse_light + ambient_light_color).view(h, w, 3).unsqueeze(0) # shape [1, H, W, 3]

        if shading == "albedo":
            rgb_output = albedo + textureless_color * 0
        elif shading == "textureless": # textureless -> no visual texture -> only lighting color
            rgb_output = albedo * 0 + textureless_color
        elif shading == "diffuse":
            rgb_output = albedo * textureless_color
        else:
            raise ValueError(f"Unknown shading mode {shading}")
            
        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]
        viewcos = rot_normal[..., [2]]

        shading_normal = shading_normal.view(-1, H, W, 3) 
        # compute shading normal in view space (for ControlNet conditioning)
        # Ref: https://github.com/threestudio-project/threestudio/blob/47e65d4593fd303da86945760fc72a89f1c3bf6f/threestudio/models/renderers/nvdiff_rasterizer.py#L66
        w2c = torch.inverse(pose[:3, :3]).unsqueeze(0) # pose shape [4,4], w2c shape [1, 3, 3]
        shading_normal_viewspace = torch.einsum('bij,bhwj->bhwi', w2c, shading_normal) # shape [1, 64, 64, 3], range [-1, 1]
        shading_normal_viewspace = F.normalize(shading_normal_viewspace, dim=-1)
        # antialias
        shading_normal_viewspace = torch.lerp(torch.zeros_like(shading_normal_viewspace), shading_normal_viewspace, mask.float())
        shading_normal_viewspace = dr.antialias(shading_normal_viewspace.contiguous(), rast, v_clip, self.mesh.f).squeeze(0) # [1, H, W, 3], range [-1, 1]


        # antialias
        rgb = dr.antialias(rgb_output, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
        rgb = alpha * rgb + (1 - alpha) * bg_color
        albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
        albedo = alpha * albedo + (1 - alpha) * bg_color
        shading_normal = dr.antialias(shading_normal, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
        perturb_normal = dr.antialias(perturb_normal, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
        # User [0, 0, 1] as bg_color, resonable but less visible
        # perturb_normal = alpha * perturb_normal + (1 - alpha) * torch.tensor([0, 0, 1], dtype=torch.float32, device=perturb_normal.device).unsqueeze(0) # [H, W, 3] 
        # Use bg_color as bg_color, more visible
        perturb_normal = alpha * perturb_normal + (1 - alpha) * bg_color # [H, W, 3]
        target_albedo = dr.antialias(target_albedo, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
        target_albedo = alpha * target_albedo + (1 - alpha) * bg_color
        
        
        target_perturb_normal = dr.antialias(target_perturb_normal, rast, v_clip, self.mesh.f).squeeze(0)
        # print(f"before computing target_shading_normal_viewspace, check target_perturb_normal {target_perturb_normal.shape} range [{target_perturb_normal.min()},{target_perturb_normal.max()}] ")  # torch.Size([512, 512, 3]) range [-1.0,1.0]
        target_shading_normal = tangent * target_perturb_normal[..., [0]] - bitangent * target_perturb_normal[..., [1]] + normal * target_perturb_normal[..., [2]] # [512, 512, 3]
        target_shading_normal = safe_normalize(target_shading_normal.view(-1, 3)) 
        target_shading_normal = target_shading_normal.view(-1, H, W, 3) # [1, 512, 512, 3]
        target_shading_normal_viewspace = torch.einsum('bij,bhwj->bhwi', w2c, target_shading_normal) 
        target_shading_normal_viewspace = F.normalize(target_shading_normal_viewspace, dim=-1)
        target_shading_normal_viewspace = torch.lerp(torch.zeros_like(target_shading_normal_viewspace), target_shading_normal_viewspace, mask.float()) # [1, 512, 512, 3]
        target_shading_normal_viewspace = dr.antialias(target_shading_normal_viewspace.contiguous(), rast, v_clip, self.mesh.f).squeeze(0) # [1, H, W, 3], range [-1, 1]
        
        target_perturb_normal = alpha * target_perturb_normal + (1 - alpha) * bg_color # [H, W, 3]

        label_map = dr.antialias(label_map, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 1]

        if self.opt.num_part_label > 0:
            target_perturb_normal2 = dr.antialias(target_perturb_normal2, rast, v_clip, self.mesh.f).squeeze(0)
            target_perturb_normal2 = alpha * target_perturb_normal2 + (1 - alpha) * bg_color
            target_shading_normal2 = tangent * target_perturb_normal2[..., [0]] - bitangent * target_perturb_normal2[..., [1]] + normal * target_perturb_normal2[..., [2]]
            target_shading_normal2 = safe_normalize(target_shading_normal2.view(-1, 3))
            target_shading_normal2 = target_shading_normal2.view(-1, H, W, 3)
            target_shading_normal_viewspace2 = torch.einsum('bij,bhwj->bhwi', w2c, target_shading_normal2)
            target_shading_normal_viewspace2 = F.normalize(target_shading_normal_viewspace2, dim=-1)
            target_shading_normal_viewspace2 = torch.lerp(torch.zeros_like(target_shading_normal_viewspace2), target_shading_normal_viewspace2, mask.float())
            target_shading_normal_viewspace2 = dr.antialias(target_shading_normal_viewspace2.contiguous(), rast, v_clip, self.mesh.f).squeeze(0) # [1, H, W, 3], range [-1, 1]

        # ssaa
        if ssaa != 1:
            rgb = scale_img_hwc(rgb, (h0, w0))
            albedo = scale_img_hwc(albedo, (h0, w0))
            alpha = scale_img_hwc(alpha, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            normal = scale_img_hwc(normal, (h0, w0))
            viewcos = scale_img_hwc(viewcos, (h0, w0))
            shading_normal = scale_img_hwc(shading_normal, (h0, w0))
            perturb_normal = scale_img_hwc(perturb_normal, (h0, w0))
            shading_normal_viewspace = scale_img_hwc(shading_normal_viewspace, (h0, w0))
            target_albedo = scale_img_hwc(target_albedo, (h0, w0))
            target_perturb_normal = scale_img_hwc(target_perturb_normal, (h0, w0))
            target_shading_normal_viewspace = scale_img_hwc(target_shading_normal_viewspace, (h0, w0))
            if self.opt.num_part_label > 0:
                target_perturb_normal2 = scale_img_hwc(target_perturb_normal2, (h0, w0))
                target_shading_normal_viewspace2 = scale_img_hwc(target_shading_normal_viewspace2, (h0, w0))
            label_map = scale_img_hwc(label_map, (h0, w0))

        results['lambertian'] = rgb.clamp(0, 1)
        results['albedo'] = albedo
        results['mask'] = alpha
        results['depth'] = depth
        results['normal'] = (normal + 1) / 2
        results['viewcos'] = viewcos
        results['perturb_normal'] = perturb_normal
        results['shading_normal'] = shading_normal
        results['shading_normal_viewspace'] = shading_normal_viewspace
        results['target_albedo'] = target_albedo
        results['target_perturb_normal'] = target_perturb_normal
        results['target_shading_normal_viewspace'] = target_shading_normal_viewspace
        if self.opt.num_part_label > 0:
            results['target_perturb_normal2'] = target_perturb_normal2
            results['target_shading_normal_viewspace2'] = target_shading_normal_viewspace2
        results['label'] = label_map

        return results
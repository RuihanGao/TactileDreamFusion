import os
import cv2
import tqdm
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from mesh_tactile import Mesh
from cam_utils import OrbitCamera
from utils import safe_normalize

import pdb

"""
Modified from https://github.com/ashawkey/kiuikit/blob/main/kiui/render.py. 
Remove GUI visualization and add more flexibility to the renderer for tactile texture mapping.
"""

class GUI:
    def __init__(self, opt):
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.bg_color = torch.ones(3, dtype=torch.float32).cuda() # default white bg

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.light_dir = np.array([0, 0])

        light_intensity = 1
        self.ambient_light_color = light_intensity * torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).cuda()
        self.directional_light_color = light_intensity * torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).cuda()
        self.ambient_light_ratio = opt.ambient_light_ratio

        # auto-rotate
        self.auto_rotate_cam = False
        self.auto_rotate_light = True
        
        # load mesh
        self.mesh = Mesh.load(opt.mesh, front_dir=opt.front_dir, opt=opt, resize=opt.resize)

        # render_mode
        self.render_modes = ['depth', 'normal', 'custom', 'shading_normal', 'tangent', 'uv', 'textureless', 'viewspace_normal', 'label_map']
        if self.mesh.albedo is not None or self.mesh.vc is not None:
            self.render_modes.extend(['albedo', 'lambertian'])
        if self.mesh.tactile_normal is not None:
            self.render_modes.extend(['tactile_normal'])

        # load pbr if enabled
        if self.opt.pbr:
            import envlight
            if self.opt.envmap is None:
                hdr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/lights/mud_road_puresky_1k.hdr')
            else:
                hdr_path = self.opt.envmap
            self.light = envlight.EnvLight(hdr_path, scale=2, device='cuda')
            self.FG_LUT = torch.from_numpy(np.fromfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/lights/bsdf_256_256.bin"), dtype=np.float32).reshape(1, 256, 256, 2)).cuda()

            self.metallic_factor = 1
            self.roughness_factor = 1

            self.render_modes.append('pbr')
        
        if opt.mode in self.render_modes:
            self.mode = opt.mode
        else:
            raise ValueError(f'[ERROR] mode {opt.mode} not supported')
            
        self.glctx = dr.RasterizeGLContext()


    def step(self):

        if not self.need_update:
            return
    
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        # do MVP for vertices
        pose = torch.from_numpy(self.cam.pose.astype(np.float32)).cuda()
        proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).cuda()
        
        v_cam = torch.matmul(F.pad(self.mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (self.H, self.W))

        # alpha is a foreground mask
        alpha = (rast[..., 3:] > 0).float()
        alpha = dr.antialias(alpha, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]

        if self.mode == "uv":
            # render the 2D uv coordinates of each 3D vertex to the image
            uv, _ = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft) # [1, 800, 800, 2], range [0, 1]
            uv = torch.where(rast[..., 3:] > 0, uv, torch.tensor(1).to(uv.device))
            # concate uv with ones, convert it from 2 channels to 3 channels
            uv = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)
            buffer = uv[0].detach().cpu().numpy()
        
        elif self.mode == 'depth':
            depth, _ = dr.interpolate(-v_cam[..., [2]], rast, self.mesh.f) # [1, H, W, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
            buffer = depth.squeeze(0).detach().cpu().numpy().repeat(3, -1) # [H, W, 3]
        elif self.mode == 'normal':
            normal, _ = dr.interpolate(self.mesh.vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
            normal = safe_normalize(normal)
            normal_image = (normal[0] + 1) / 2
            normal_image = torch.where(rast[..., 3:] > 0, normal_image, torch.tensor(1).to(normal_image.device)) # remove background
            buffer = normal_image.squeeze(0).detach().cpu().numpy()
       
        elif self.mode == 'tactile_normal' or self.mode == 'shading_normal' or self.mode == 'tangent' or self.mode == 'viewspace_normal':

            if self.mesh.tactile_normal is not None:
                texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
                perturb_normal = dr.texture(self.mesh.tactile_normal.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear', max_mip_level=5).to(torch.float32) # [1, H, W, 3], range [-1, 1], H=W=800 # self.mesh.tactile_normal [1024, 1024, 3]
      
            else:
                # create a map of [0, 0, 1]
                perturb_normal = torch.tensor([0, 0, 1], dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(rast.device)
                # repeat to shape [1, H, W, 3]
                perturb_normal = perturb_normal.repeat(1, self.H, self.W, 1)

            if self.mode == 'tactile_normal':
                # convert range [-1, 1] to [0, 1]
                perturb_normal = perturb_normal*0.5 + 0.5
                perturb_normal = torch.where(rast[..., 3:] > 0, perturb_normal, torch.tensor(1).to(perturb_normal.device))
                buffer = perturb_normal[0].detach().cpu().numpy()
            else:
                # compute normal
                normal, _ = dr.interpolate(self.mesh.vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
                normal = safe_normalize(normal)
                # compute tangent
                tangent, _ = dr.interpolate(self.mesh.v_tangent.unsqueeze(0).contiguous(), rast, self.mesh.fn)
                tangent = safe_normalize(tangent) # shape [1, H, W, 3]
                if self.mode == 'tangent':
                    tangent = tangent*0.5 + 0.5 # range [-1, 1] -> [0, 1]
                    tangent  = torch.where(rast[..., 3:] > 0, tangent, torch.tensor(1).to(tangent.device))
                    buffer = tangent[0].detach().cpu().numpy()
                else:
                    # shadind_normal or viewspace_normal
                    bitangent = safe_normalize(torch.cross(tangent, normal, dim=-1))
                    # compute shading normal
                    shading_normal = tangent * perturb_normal[..., [0]] - bitangent * perturb_normal[..., [1]] + normal * perturb_normal[..., [2]] # shape [1, H, W, 3]
                    if self.mode == 'shading_normal':
                        shading_normal = safe_normalize(shading_normal)
                        shading_normal = shading_normal*0.5 + 0.5 # range [-1, 1] -> [0, 1]
                        shading_normal = torch.where(rast[..., 3:] > 0, shading_normal, torch.tensor(1).to(shading_normal.device))
                        buffer = shading_normal[0].detach().cpu().numpy()
                    else:
                        # viewspace_normal
                        mask = (rast[0, ..., 3:] > 0).float().unsqueeze(0) 
                        w2c = torch.inverse(pose[:3, :3]).unsqueeze(0) # [1, 3, 3]
                        shading_normal_viewspace = torch.einsum('bij,bhwj->bhwi', w2c, shading_normal) # [1, H, W, 3]
                        shading_normal_viewspace = shading_normal_viewspace * 0.5 + 0.5  # range [-1, 1] -> [0, 1]
                        shading_normal_viewspace = torch.where(rast[..., 3:] > 0, shading_normal_viewspace, torch.tensor(1).to(shading_normal_viewspace.device)) # change background to white
                        buffer = shading_normal_viewspace[0].detach().cpu().numpy()


        elif self.mode == 'label_map':
            # render the label map of each vertex to the image       
            if self.mesh.label_map is not None:
                texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
                label_map = dr.texture(self.mesh.label_map.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear', max_mip_level=5).to(torch.float32) # [1, H, W, 3], range [0, 1]
                label_map = torch.where(rast[..., 3:] > 0, label_map, torch.tensor(1).to(label_map.device))
                buffer = label_map[0].detach().cpu().numpy() # [800, 800, 3]
            else:
                label_map = torch.ones_like(rast[..., :1])

        
        elif self.mode == 'custom':
            assert self.opt.map_path is not None, "custom map path not found."
            # Load the texture map
            custom_texture = cv2.imread(self.opt.map_path, cv2.IMREAD_UNCHANGED) # The input map should be in range [0, 255]
            if custom_texture.shape[-1] == 4: # RGBA
                custom_texture = custom_texture[..., :3]
            if custom_texture.shape[-1] == 3:
                custom_texture = cv2.cvtColor(custom_texture, cv2.COLOR_BGR2RGB)
            custom_texture = custom_texture.astype(np.float32) / 255.0
            custom_texture = torch.from_numpy(custom_texture).unsqueeze(0).cuda() # [1, H, W, C]
            texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
            custom_map = dr.texture(custom_texture, texc, filter_mode='linear')
            custom_map = torch.where(rast[..., 3:] > 0, custom_map, torch.tensor(0).to(custom_map.device)) # remove background
            buffer = custom_map[0].detach().cpu().numpy()


        else: # color (albedo)
            texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
            if self.mesh.vc is not None: # use vertex color if exists 
                albedo, _ = dr.interpolate(self.mesh.vc.unsqueeze(0).contiguous(), rast, self.mesh.f)
            # use texture image
            else:
                albedo = dr.texture(self.mesh.albedo.unsqueeze(0), texc, filter_mode='linear') # [1, H, W, 3]
            albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device)) # remove background

            if self.mode == 'albedo':
                albedo = albedo * alpha + self.bg_color * (1 - alpha)
                buffer = albedo[0].detach().cpu().numpy()
            else:
                normal, _ = dr.interpolate(self.mesh.vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
                normal = safe_normalize(normal)

                if self.mesh.tactile_normal is None:
                    shading_normal = normal
                else:
                    # take both mesh normal and tactile normal into consideration for shading
                    perturb_normal = dr.texture(self.mesh.tactile_normal.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear', max_mip_level=5).to(torch.float32) # [1, H, W, 3]
                    # compute tangent
                    tangent, _ = dr.interpolate(self.mesh.v_tangent.unsqueeze(0).contiguous(), rast, self.mesh.fn)
                    tangent = safe_normalize(tangent[0]).unsqueeze(0) # shape [1, H, W, 3]

                    bitangent = safe_normalize(torch.cross(tangent, normal, dim=-1))
                    shading_normal = tangent * perturb_normal[..., [0]] - bitangent * perturb_normal[..., [1]] + normal * perturb_normal[..., [2]] # shape [1, H, W, 3]
                    shading_normal = safe_normalize(shading_normal)
                
                if self.mode == 'lambertian' or self.mode == 'textureless':
                    light_d = np.deg2rad(self.light_dir)
                    light_d = np.array([
                        np.cos(light_d[0]) * np.sin(light_d[1]),
                        -np.sin(light_d[0]),
                        np.cos(light_d[0]) * np.cos(light_d[1]),
                    ], dtype=np.float32)
                    light_d = torch.from_numpy(light_d).to(albedo.device)
                    textureless_color = self.ambient_light_ratio * self.ambient_light_color.unsqueeze(1).unsqueeze(2) + (1 - self.ambient_light_ratio) * self.directional_light_color.unsqueeze(1).unsqueeze(2) * (shading_normal @ light_d).clamp(min=0)

                    if self.mode == 'lambertian':
                        # apply background mask (alpha)
                        albedo = (albedo * textureless_color.unsqueeze(-1)) * alpha + self.bg_color * (1 - alpha)
                        buffer = albedo[0].detach().cpu().numpy()
                    else:
                        # textureless
                        # apply background mask (alpha)
                        textureless_color = textureless_color.unsqueeze(-1) * alpha + self.bg_color * (1 - alpha)
                        buffer = textureless_color[0].detach().cpu().numpy() # [800, 800, 3]
                    
                elif self.mode == 'pbr':
                    if self.mesh.metallicRoughness is not None:
                        metallicRoughness = dr.texture(self.mesh.metallicRoughness.unsqueeze(0), texc, filter_mode='linear') # [1, H, W, 3]
                        metallic = metallicRoughness[..., 2:3] * self.metallic_factor
                        roughness = metallicRoughness[..., 1:2] * self.roughness_factor
                    else:
                        metallic = torch.ones_like(albedo[..., :1]) * self.metallic_factor
                        roughness = torch.ones_like(albedo[..., :1]) * self.roughness_factor

                    xyzs, _ = dr.interpolate(self.mesh.v.unsqueeze(0), rast, self.mesh.f) # [1, H, W, 3]
                    viewdir = safe_normalize(xyzs - pose[:3, 3])

                    n_dot_v = (shading_normal * viewdir).sum(-1, keepdim=True) # [1, H, W, 1]
                    reflective = n_dot_v * shading_normal * 2 - viewdir

                    diffuse_albedo = (1 - metallic) * albedo

                    fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1) # [H, W, 2]
                    fg = dr.texture(
                        self.FG_LUT,
                        fg_uv.reshape(1, -1, 1, 2).contiguous(),
                        filter_mode="linear",
                        boundary_mode="clamp",
                    ).reshape(1, self.H, self.W, 2)
                    F0 = (1 - metallic) * 0.04 + metallic * albedo
                    specular_albedo = F0 * fg[..., 0:1] + fg[..., 1:2]

                    diffuse_light = self.light(shading_normal)
                    specular_light = self.light(reflective, roughness)

                    color = diffuse_albedo * diffuse_light + specular_albedo * specular_light # [H, W, 3]
                    color = color * alpha + self.bg_color * (1 - alpha)

                    buffer = color[0].detach().cpu().numpy()
                    

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.render_buffer = buffer
        self.need_update = False

        if self.auto_rotate_cam:
            self.cam.orbit(5, 0)
            self.need_update = True
        
        if self.auto_rotate_light:
            self.light_dir[1] += 1
            self.need_update = True

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh', type=str, help="path to mesh (obj, ply, glb, ...)")
    parser.add_argument('--pbr', action='store_true', help="enable PBR material")
    parser.add_argument('--envmap', type=str, default=None, help="hdr env map path for pbr")
    parser.add_argument('--front_dir', type=str, default='+z', help="mesh front-facing dir")
    parser.add_argument('--mode', default='albedo', type=str, choices=['uv', 'lambertian', 'albedo', 'normal', 'depth', 'pbr', 'tactile_normal', 'shading_normal', 'tangent', 'custom', 'textureless', 'viewspace_normal', 'label_map'], help="rendering mode, add custom mode to render a specific texture map.")
    parser.add_argument('--map_path', type=str, default=None, help="path to custom texture map, only used if mode is 'custom'")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--save', type=str, default=None, help="path to save example rendered images")
    parser.add_argument('--elevation', type=int, default=0, help="rendering elevation")
    parser.add_argument('--azimuth', type=int, default=None, help="rendering azimuth for image rendering. If None, num_azimuth images will be rendered, evenly distributed from 0 to 360.") 
    parser.add_argument('--light_rel_ele', type=int, default=0, help="relative elevation of light to camera")
    parser.add_argument('--light_rel_azi', type=int, default=0, help="relative azimuth of light to camera")
    parser.add_argument('--num_azimuth', type=int, default=8, help="number of images to render from different azimuths")
    parser.add_argument('--num_elevation', type=int, default=None, help="number of images to render from different azimuths")
    parser.add_argument('--save_video', type=str, default=None, help="path to save rendered video")
    parser.add_argument('--no_tactile', action="store_true")
    parser.add_argument('--static_object', action="store_true")
    parser.add_argument('--no_auto_resize', action="store_true", help="option to not resize the mesh")
    parser.add_argument('--ambient_light_ratio', type=float, default=0.1, help="ambient light ratio")

    opt = parser.parse_args()
    opt.resize = not opt.no_auto_resize
    print(f"check option to resize mesh: {opt.resize}")

    gui = GUI(opt)

    if opt.save is not None:
        os.makedirs(opt.save, exist_ok=True)
        # render from fixed views and save all images
        if opt.num_elevation is not None:
            elevation = np.linspace(-20, 20, opt.num_elevation, dtype=np.int32)
        else:
            elevation = [opt.elevation,]
        if opt.azimuth is not None:
            azimuth = [opt.azimuth]
        else:
            # render from 0 to 360, evenly distributed to obtain opt.num_azimuth images
            azimuth = np.linspace(0, 360, opt.num_azimuth, dtype=np.int32, endpoint=False)
        for ele in tqdm.tqdm(elevation):
            for azi in tqdm.tqdm(azimuth):
                gui.cam.from_angle(ele, azi)
                light_ele = ele + opt.light_rel_ele
                light_azi = (azi + opt.light_rel_azi) % 360
                gui.light_dir = np.array([light_ele, light_azi])
                gui.need_update = True
                gui.step()
                image = (gui.render_buffer * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(opt.save, f'{ele}_{azi}_light_{light_ele}_{light_azi}_{opt.ambient_light_ratio}_{opt.mode}.png'), image)
    elif opt.save_video is not None:
        import imageio
        elevation = [opt.elevation,]

        if opt.static_object:
            # fix the object, only rotate the light
            for fix_azi in [0, 180]:
                # render two videos, front and back
                images = []
                azimuth = [fix_azi] * 360 # for static rendering
                for ele in tqdm.tqdm(elevation):
                    for azi in tqdm.tqdm(azimuth):
                        gui.cam.from_angle(ele, azi)
                        # gui.light_dir = np.array([ele, azi]) # light will follow camera to rotate
                        gui.need_update = True
                        gui.step()
                        image = (gui.render_buffer * 255).astype(np.uint8)
                        images.append(image)
                images = np.stack(images, axis=0) # [N, H, W, C]
                # ~4 seconds, 120 frames at 30 fps
                os.makedirs(os.path.dirname(opt.save_video), exist_ok=True)
                video_output_path = opt.save_video.replace('.mp4', f'_{fix_azi}.mp4')
                imageio.mimwrite(video_output_path, images, fps=30, quality=8, macro_block_size=1)
        
        else:
            # rotate the object
            images = []
            azimuth = np.arange(0, 360, 1, dtype=np.int32) # front-->back-->front

            for ele in tqdm.tqdm(elevation):
                for azi in tqdm.tqdm(azimuth):
                    gui.cam.from_angle(ele, azi)
                    gui.need_update = True
                    gui.step()
                    image = (gui.render_buffer * 255).astype(np.uint8)
                    images.append(image)
            images = np.stack(images, axis=0) # [N, H, W, C]
            # ~4 seconds, 120 frames at 30 fps
            os.makedirs(os.path.dirname(opt.save_video), exist_ok=True)
            imageio.mimwrite(opt.save_video, images, fps=30, quality=8, macro_block_size=1)

    else:
        gui.render()


if __name__ == '__main__':
    main()
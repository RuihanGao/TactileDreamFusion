import os
import cv2
import time
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

import rembg
import json
import os.path as osp
import torchvision

import argparse
from omegaconf import OmegaConf

from cam_utils import orbit_camera, OrbitCamera, undo_orbit_camera
from mesh_renderer_tactile import Renderer

from utils import convert_images_to_video, toggle_variable
import lpips
import imageio
from seg.diff_seg import seg_attn
from utils import generate_textured_prompt
import pdb

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.patch_cam = OrbitCamera(opt.patch_W, opt.patch_H, r=opt.patch_radius, fovy=opt.patch_fovy, proj_mode=opt.patch_cam_proj_mode, view_volume_size=opt.view_volume_size)
        self.seed_everything("random")

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        # For diffusion loss
        self.guidance_normalcontrolnet = None
        self.guidance_tactile = None
        
        # renderer
        self.renderer = Renderer(opt).to(self.device)

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.lpips_loss = lpips.LPIPS(net="vgg").to(self.device)
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        self.vis_modes = ["controlnet_refined_images", "controlnet_control_images", "rendered_lambertians", "rendered_albedos", "rendered_target_albedos", "rendered_perturb_normals", "rendered_target_perturb_normals", "rendered_guidance_perturb_normals", "rendered_target_perturb_normal2s", "rendered_guidance_perturb_normal2s", "rendered_albedos_patch", "rendered_target_albedos_patch", "rendered_labels", "rendered_labels_patch", "rendered_masks", "rendered_masks_patch", "seg_masks_partA", "seg_masks_partB", "seg_masks_partA_rendered", "seg_masks_partB_rendered", "seg_masks_partA_rendered_patch", "seg_masks_partB_rendered_patch"]
        print(f"Initialized GUI, prompt: {self.prompt}, negative_prompt: {self.negative_prompt}")



    def seed_everything(self, seed="random"):
        try:
            seed = int(seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    
    def prepare_train(self):
        self.step = 0
        # setup training
        self.optimizer = torch.optim.Adam(self.renderer.get_params())

        # lazy load guidance model
        if self.guidance_normalcontrolnet is None and self.opt.lambda_normalcontrolnet > 0:
            # Normal-conditioned ControlNet with multi-step denoising. It takes normal rendering as a control image and the lambertian rendering as an input image
            from guidance.controlnet_utils import ControlNet
            self.guidance_normalcontrolnet = ControlNet(device=self.device)
            
            with torch.no_grad():
                self.guidance_normalcontrolnet.get_text_embeds([self.prompt], [self.negative_prompt])
            print(f"[INFO] loaded ControlNetPipe(Normal)!")


        if self.guidance_tactile is None and self.opt.lambda_tactile_guidance > 0:
            # Texture Dreambooth with multi-step denoising. It is used to refine rendered tactile patches.
            from guidance.tactile_guidance_utils import TextureDreambooth
            if self.opt.tactile_lora_dir is None:
                self.opt.tactile_lora_dir = f"lora_{self.opt.tactile_texture_object.lower()}_sks"
            print(f"[INFO] loading TactileGuidance from lora dir {self.opt.tactile_lora_dir}")
            tactile_lora_parent_dir = "TextureDreambooth/output/"
            self.guidance_tactile = TextureDreambooth(device=self.device, fp16=False, sd_version="1.4", lora_dir=osp.join(tactile_lora_parent_dir, self.opt.tactile_lora_dir))
            # the TextureDreambooth is trained per texture with "sks normal map" as the input text
            self.guidance_tactile.get_text_embeds(["sks normal map"], [""])

            if self.opt.num_part_label > 0:
                # load a second Texture Dreambooth for multi-part texture generation
                self.opt.tactile_lora_dir_partB = f"lora_{self.opt.texture2_name.lower()}_sks"
                print(f"[INFO] loading 2nd TactileGuidance from lora dir {self.opt.tactile_lora_dir_partB}")
                self.guidance_tactile_partB = TextureDreambooth(device=self.device, fp16=False, sd_version="1.4", lora_dir=osp.join(tactile_lora_parent_dir, self.opt.tactile_lora_dir_partB))
                self.guidance_tactile_partB.get_text_embeds(["sks normal map"], [""])
            
            print(f"[INFO] loaded TactileGuidance!")

    
    def train_step(self, return_loss_dict=False, iter_idx=0):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        albedo_recon_loss_func = torch.nn.MSELoss()
        albedo_regularization_loss_func = torch.nn.MSELoss()
        label_field_loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

        loss_dict = {}

        step_vis_dict = {k: [] for k in self.vis_modes}

        self.step += 1
        # compute the step_ratio to adjust the strength of guidance as the training progresses
        if self.opt.iters_refine > self.opt.iters_init: 
            step_ratio = min(1, (self.step - self.opt.iters_init) / (self.opt.iters_refine - self.opt.iters_init))

        loss = 0

        ### novel view (manual batch)
        render_resolution = 512
        poses = []
        vers, hors, radii = [], [], []
        seg_masks = None

        # collect all rendered views to compute texture loss
        batch_vis_modes = ["rendered_perturb_normals",   "rendered_target_perturb_normals", "rendered_albedos", "rendered_target_albedos", "rendered_lambertians", "rendered_labels", "rendered_masks", "rendered_target_albedos_patch", "rendered_albedos_patch", "rendered_labels_patch",  "rendered_masks_patch", "rendered_target_perturb_normal2s", "rendered_guidance_perturb_normals", "rendered_target_albedos",   "rendered_guidance_normals",   "rendered_shading_normal_viewspaces", "rendered_guidance_perturb_normal2s",]
        batch_vis_dict = {k: [] for k in batch_vis_modes}
    

        # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
        max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)


        # rendering loss
        # render tactile patches for close-up view
        patch_render_resolution = 512
        patch_poses = []
        # print(f"Render tactile patches for close-up view ...")
        ssaa = 1
        for patch_idx in range(self.opt.patch_batch_size):
            # Sample tactile patches by mesh vertices and vertex normals
            total_num_vertices = self.renderer.mesh.v.shape[0]
            # sample a mesh vertex and obtain its normal
            vertex_idx = np.random.randint(0, total_num_vertices)
            sampled_vertex = self.renderer.mesh.v[vertex_idx] # cuda tensor [3]
            sampled_normal = self.renderer.mesh.vn[vertex_idx]
            cam_pose = sampled_vertex + self.opt.patch_cam_dist * sampled_normal # NOTE: the cam_dist doesn't make a difference for orthographic projection
            # convert the 3D coordinate to a 4x4 matrix, where the last column is the translation
            cam_pose_T = np.eye(4)
            cam_pose_T[:3, 3] = cam_pose.cpu().numpy()
            # convert the camera pose to radius, elevation, azimuth
            elevation, azimuth, radius = undo_orbit_camera(cam_pose_T, is_degree=True)
            # NOTE: set the camera to look at the sampled vertex instead of the origin so that the projection is aligned with sampled normal
            patch_pose = orbit_camera(elevation, azimuth, radius, target=sampled_vertex.cpu().numpy(), customize_pos=True) 
            patch_poses.append(patch_pose)
            # render the patch (only normal map)
            patch_out = self.renderer.render(patch_pose, self.patch_cam.perspective, patch_render_resolution, patch_render_resolution, ssaa=ssaa)

            for vis_mode in ["rendered_perturb_normals", "rendered_target_perturb_normals", "rendered_target_albedos_patch", "rendered_masks_patch", "rendered_labels_patch", "rendered_albedos_patch", "rendered_target_perturb_normal2s"]:
                
                if "_patch" in vis_mode:
                    out_vis_mode_str = f"{vis_mode[9:-7]}" # remove "rendered_", "s_patch"
                else:
                    out_vis_mode_str = vis_mode[9:-1]
                    out_vis_mode_str = out_vis_mode_str.replace("perturb_normal", "shading_normal_viewspace")
                if out_vis_mode_str in patch_out and len(patch_out[out_vis_mode_str])>0:
                    batch_vis_dict[vis_mode].append(patch_out[out_vis_mode_str].permute(2,0,1).contiguous().unsqueeze(0))

        
        for _ in range(self.opt.batch_size):
            # Sample camera views by randomizing the radius, elevation, and azimuth
            ver = np.random.randint(min_ver, max_ver)
            hor = np.random.randint(-180, 180)
            radius = np.random.uniform(self.opt.radius_range[0], self.opt.radius_range[1])

            vers.append(ver)
            hors.append(hor)
            radii.append(radius)

            pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
            poses.append(pose)

            # random render resolution
            ssaa = min(self.opt.ssaa_max_scale, max(self.opt.ssaa_min_scale, self.opt.ssaa_max_scale * np.random.random()))
            out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

            # collect all rendered views to compute texture loss, [H, W, C] -> [1, C, H, W]
            for vis_mode in ["rendered_masks", "rendered_albedos", "rendered_target_albedos", "rendered_lambertians", "rendered_labels", "rendered_shading_normal_viewspaces"]:
                out_vis_mode_str = vis_mode[9:-1] # remove "rendered_" and "s"
                batch_vis_dict[vis_mode].append(out[out_vis_mode_str].permute(2,0,1).contiguous().unsqueeze(0))

        poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)
        for k, k_list in batch_vis_dict.items():
            if len(k_list) > 0:
                batch_vis_dict[k] = torch.cat(k_list, dim=0)


        # change the background to white
        bg_mask = torch.all(batch_vis_dict["rendered_masks"] == 0, dim=1).unsqueeze(1) # shape [N, H, W] -> [N, 1, H, W]
        fg_mask = 1 - bg_mask.to(torch.float32) # [N, 1, 512, 512]
        bg_mask_patch = torch.all(batch_vis_dict["rendered_masks_patch"] == 0, dim=1).unsqueeze(1) # shape [N, H, W] -> [N, 1, H, W]
        fg_mask_patch = 1 - bg_mask_patch.to(torch.float32)
        
            
        ##### Label field loss #####
        if self.opt.num_part_label > 0:
            # multi-part segmentation

            # obtain the target labels from the attention maps
            self_attn_list, cross_attn_list = self.guidance_normalcontrolnet.refine(pred_rgb=batch_vis_dict["rendered_target_albedos"], control_images=None, guidance_scale=100, steps=50, strength=0.8, controlnet_conditioning_scale=0, return_attn=True)                                 
            token_indices = [self.opt.partA_idx, self.opt.partB_idx]
            seg_masks = seg_attn(self_attn_list, cross_attn_list, token_indices, fg_mask=fg_mask.squeeze(1).squeeze(0)).unsqueeze(0) # [1, 2, 512, 512]
            seg_masks = seg_masks * fg_mask # [1, 2, 512, 512], range [0, 1], grad False
            partA_mask = seg_masks[:, 0:1] # [1, 1, 512, 512]
            partB_mask = seg_masks[:, 1:2] # [1, 1, 512, 512]
            # For cross-entropy, the target can be either containing classs indices or containing class probabilities [0,1] and same shape as input.
            # we follow the class probabilities [0,1] setting and concatenate partA_mask and partB_mask to form the target, i.e. seg_masks

            # if one of seg_masks is the same as fg_mask, the segmentation fails, thus skip this iteration
            if torch.all(seg_masks[:,0] == fg_mask[:,0]) or torch.all(seg_masks[:,1] == fg_mask[:,0]):
                print(f"[WARN] iter {iter_idx}, the segmentation fails, skip the label field loss computation")
                loss_label_field = torch.tensor(0.0).to(self.device)
                partA_mask = torch.zeros_like(partA_mask)
                partB_mask = torch.zeros_like(partB_mask)
            else:
                # flatten the seg_masks to compute only the foreground region
                seg_masks_flattened = seg_masks.view(2, -1) # [2, 512*512]
                fg_mask_flattened = fg_mask.view(1, -1) # [1, 512*512]
                seg_masks_selected = seg_masks_flattened[:, fg_mask_flattened[0] > 0] # [2, num_fg_pixels]
                rendered_labels_flattened = batch_vis_dict["rendered_labels"].view(2, -1) # [2, 512*512]
                rendered_labels_selected = rendered_labels_flattened[:, fg_mask_flattened[0] > 0] # [2, num_fg_pixels], grad True

                # compute cross-entropy loss to train the label field
                loss_label_field = label_field_loss_func(rendered_labels_selected.permute(1, 0), seg_masks_selected.permute(1, 0)) * self.opt.lambda_label_field # input and target shape [num_pixels, num_classes]

    
            # 20241026: partA_mask_rendered and partB_mask_rendered are used for visualization only. so we don't need to track gradients on them. take the max value of the last channel to obtain the part mask
            # take the max value of the last channel to obtain the part mask
            predicted_labels = torch.argmax(batch_vis_dict["rendered_labels"], dim=1).unsqueeze(1) # [1, 512, 512], grad False
            partA_mask_rendered = (predicted_labels == 0).float() * fg_mask # [1, 1, 512, 512]
            partB_mask_rendered = (predicted_labels == 1).float() * fg_mask # [1, 1, 512, 512]
            # concate partA_mask and partB_mask to form the rendered seg_masks
            predicted_labels = torch.cat([partA_mask_rendered, partB_mask_rendered], dim=1).float() # [1, 2, 512, 512], grad False

            # convert the binary mask to part1 and part2 mask
            predicted_labels_patch = torch.argmax(batch_vis_dict["rendered_labels_patch"], dim=1).unsqueeze(1) # [4, 512, 512], grad False
            partA_mask_patch_rendered = (predicted_labels_patch == 0).float()
            partB_mask_patch_rendered = (predicted_labels_patch == 1).float()
            # multiply each part mask with the foreground mask
            partA_mask_patch_rendered = partA_mask_patch_rendered * fg_mask_patch  # [4, 1, 512, 512]
            partB_mask_patch_rendered = partB_mask_patch_rendered * fg_mask_patch # [4, 1, 512, 512]

        else:
            loss_label_field = torch.tensor(0.0).to(self.device)
        
        if "label_field" not in loss_dict:
            loss_dict["label_field"] = [loss_label_field.item()]
        else:
            loss_dict["label_field"].append(loss_label_field.item())
        loss += loss_label_field


        ##### RGB loss #####
        if iter_idx <= self.opt.iters_init:
            if self.opt.lambda_albedo_recon > 0:
                loss_albedo_regularization = self.opt.lambda_albedo_recon * albedo_recon_loss_func(batch_vis_dict["rendered_albedos"], batch_vis_dict["rendered_target_albedos"])
            else:
                loss_albedo_regularization = torch.tensor(0.0).to(self.device)
                
        else:
            # refinment stage. rename reconstruction loss to regularization loss
            if self.opt.lambda_albedo_regularization > 0:
                if hasattr(self.opt, "albedo_regularization_use_mean") and self.opt.albedo_regularization_use_mean:
                    # compute the mean color fo the rendered albedo and target albedo
                    mean_rendered_albedo = batch_vis_dict["rendered_albedos"].mean(dim=(2,3), keepdim=True)
                    mean_target_albedo = batch_vis_dict["rendered_target_albedos"].mean(dim=(2,3), keepdim=True)
                    loss_albedo_regularization = self.opt.lambda_albedo_regularization * F.mse_loss(mean_rendered_albedo, mean_target_albedo)
                else:
                    loss_albedo_regularization = self.opt.lambda_albedo_regularization * albedo_regularization_loss_func(batch_vis_dict["rendered_albedos"], batch_vis_dict["rendered_target_albedos"])
            else:
                loss_albedo_regularization = torch.tensor(0.0).to(self.device)

        if "albedo_regularization" not in loss_dict:
            loss_dict["albedo_regularization"] = [loss_albedo_regularization.item()]
        else:
            loss_dict["albedo_regularization"].append(loss_albedo_regularization.item())
        loss += loss_albedo_regularization


        ##### Tactile loss #####
        if self.opt.lambda_tactile_regularization > 0 or (self.opt.lambda_tactile_regularization_init > 0 and iter_idx <= self.opt.iters_init):

            if iter_idx < self.opt.iters_init:
                lambda_tactile_regularization = self.opt.lambda_tactile_regularization_init
            else:
                lambda_tactile_regularization = self.opt.lambda_tactile_regularization
                
            if self.opt.num_part_label == 0:
                # compute the arc cosine of the dot product of the perturb normal and target perturb normal
                loss_tactile_regularization = lambda_tactile_regularization * (1 - F.cosine_similarity(batch_vis_dict["rendered_perturb_normals"], batch_vis_dict["rendered_target_perturb_normals"]).mean())

            else:
                # compute per-part tactile regularization loss

                # compute the loss for each part and sum them up
                # use "patch_out" to obtain rendered masks
                rendered_perturb_normals_partA = batch_vis_dict["rendered_perturb_normals"] * partA_mask_patch_rendered
                rendered_target_perturb_normals_partA = batch_vis_dict["rendered_target_perturb_normals"] * partA_mask_patch_rendered
                rendered_perturb_normals_partB = batch_vis_dict["rendered_perturb_normals"] * partB_mask_patch_rendered
                batch_vis_dict["rendered_target_perturb_normal2s"] = batch_vis_dict["rendered_target_perturb_normal2s"] * partB_mask_patch_rendered

                loss_tactile_regularization_partA = self.opt.lambda_tactile_regularization_partA * (1 - F.cosine_similarity(rendered_perturb_normals_partA, rendered_target_perturb_normals_partA).mean())
                loss_tactile_regularization_partB = self.opt.lambda_tactile_regularization_partB * (1 - F.cosine_similarity(rendered_perturb_normals_partB, batch_vis_dict["rendered_target_perturb_normal2s"]).mean())
                    
                loss_tactile_regularization = lambda_tactile_regularization * (loss_tactile_regularization_partA + loss_tactile_regularization_partB)

            if "tactile_regularization" not in loss_dict:
                loss_dict["tactile_regularization"] = [loss_tactile_regularization.item()]
            else:
                loss_dict["tactile_regularization"].append(loss_tactile_regularization.item())
            loss += loss_tactile_regularization
        

        if self.opt.lambda_tactile_guidance > 0 and iter_idx > self.opt.iters_init:
            # guidance loss. use refined images instead of target images as supervision signal
            if self.opt.tacitle_guidance_mode == "multistep":
                if self.opt.num_part_label == 0:
                    # NOTE: "refine" takes image input in range [0,1] and output refined_images in range [0,1]
                    tactile_guidance_refined_images = self.guidance_tactile.refine(pred_rgb=(batch_vis_dict["rendered_perturb_normals"]+1)/2, guidance_scale=self.opt.tactile_guidance_scale, steps=self.opt.tactile_guidance_multistep_steps, strength=self.opt.tactile_guidance_multistep_strength)
                    tactile_guidance_refined_images = tactile_guidance_refined_images * 2 - 1 # convert range from [0, 1] to [-1, 1]
                    # compute image-space loss
                    loss_tactile_guidance = F.mse_loss(batch_vis_dict["rendered_perturb_normals"], tactile_guidance_refined_images) * self.opt.lambda_tactile_guidance
                    batch_vis_dict["rendered_guidance_perturb_normals"] = tactile_guidance_refined_images
                else:
                    # multi-part segmentation
                    # render the label map and use it for masking different parts
                    rendered_perturb_normals_partA = batch_vis_dict["rendered_perturb_normals"] * partA_mask_patch_rendered
                    rendered_target_perturb_normals_partA = batch_vis_dict["rendered_target_perturb_normals"] * partA_mask_patch_rendered
                    rendered_perturb_normals_partB = batch_vis_dict["rendered_perturb_normals"] * partB_mask_patch_rendered
                    batch_vis_dict["rendered_target_perturb_normal2s"] = batch_vis_dict["rendered_target_perturb_normal2s"] * partB_mask_patch_rendered

                    tactile_guidance_refined_images_partA = self.guidance_tactile.refine(pred_rgb=(rendered_perturb_normals_partA+1)/2, guidance_scale=self.opt.tactile_guidance_scale, steps=self.opt.tactile_guidance_multistep_steps, strength=self.opt.tactile_guidance_multistep_strength)
                    tactile_guidance_refined_images_partA = tactile_guidance_refined_images_partA * 2 - 1 # convert range from [0, 1] to [-1, 1]
                    # compute image-space loss
                    loss_tactile_guidance_partA = F.mse_loss(rendered_perturb_normals_partA, tactile_guidance_refined_images_partA) * self.opt.lambda_tactile_guidance
                    batch_vis_dict["rendered_guidance_perturb_normals"] = tactile_guidance_refined_images_partA
                  
                    tactile_guidance_refined_images_partB = self.guidance_tactile_partB.refine(pred_rgb=(rendered_perturb_normals_partB+1)/2, guidance_scale=self.opt.tactile_guidance_scale, steps=self.opt.tactile_guidance_multistep_steps, strength=self.opt.tactile_guidance_multistep_strength)
                    tactile_guidance_refined_images_partB = tactile_guidance_refined_images_partB * 2 - 1 # convert range from [0, 1] to [-1, 1]
                    loss_tactile_guidance_partB = F.mse_loss(rendered_perturb_normals_partB, tactile_guidance_refined_images_partB) * self.opt.lambda_tactile_guidance
                    batch_vis_dict["rendered_guidance_perturb_normal2s"]= tactile_guidance_refined_images_partB

                    loss_tactile_guidance= loss_tactile_guidance_partA + loss_tactile_guidance_partB  

            else:
                raise NotImplementedError("single-step tactile guidance is not supported for now")
  
            if "tactile_guidance" not in loss_dict:
                loss_dict["tactile_guidance"] = [loss_tactile_guidance.item()]
            else:
                loss_dict["tactile_guidance"].append(loss_tactile_guidance.item())

            loss = loss + loss_tactile_guidance


        if iter_idx >= self.opt.iters_init: 
            strength = self.opt.sd_guidance_strength + step_ratio * (self.opt.max_guidance_strength-self.opt.sd_guidance_strength) # strength -> [0, 1]
            controlnet_control_images = batch_vis_dict["rendered_shading_normal_viewspaces"].detach() # [N, 3, H, W], range [-1, 1]
            # create a variable controlnet_conditioning_toggle to toggle whether we add controlnet conditioning to refinement
            controlnet_conditioning_toggle = toggle_variable(probability=self.opt.controlnet_toggle_prob) # randomly toggle between 0 and 1
            controlnet_conditioning_scale=controlnet_conditioning_toggle*self.opt.controlnet_conditioning_scale

            # multi-step denoising 
            controlnet_refined_images, controlnet_control_images = self.guidance_normalcontrolnet.refine(pred_rgb=batch_vis_dict["rendered_lambertians"], control_images=controlnet_control_images, guidance_scale=self.opt.denoising_guidance_scale, steps=50, strength=strength, controlnet_conditioning_scale=controlnet_conditioning_scale) 
            # controlnet_refined_images: shape [1, 3, 512, 512], range [0, 1]
            # controlnet_control_images: shape [1, 3, 512, 512], range [0, 1]

            # resize the controlnet_refined_images to the same size as rendered images
            controlnet_refined_images = F.interpolate(controlnet_refined_images.float(), (render_resolution, render_resolution), mode="bilinear", align_corners=False)
            bg_tensor = torch.ones_like(controlnet_refined_images) # shape [N, 3, 512, 512]
            controlnet_refined_images = torch.where(bg_mask, bg_tensor, controlnet_refined_images)
            

            loss_normalcontrolnet_L1 = F.l1_loss(batch_vis_dict["rendered_lambertians"], controlnet_refined_images) * self.opt.lambda_normalcontrolnet_L1
            loss_normalcontrolnet_lpips = self.lpips_loss(batch_vis_dict["rendered_lambertians"], controlnet_refined_images).mean() * self.opt.lambda_normalcontrolnet_lpips
            
            if "loss_normalcontrolnet_L1" not in loss_dict:
                loss_dict["loss_normalcontrolnet_L1"] = [loss_normalcontrolnet_L1.item()]
            else:
                loss_dict["loss_normalcontrolnet_L1"].append(loss_normalcontrolnet_L1.item())
            if "loss_normalcontrolnet_lpips" not in loss_dict:
                loss_dict["loss_normalcontrolnet_lpips"] = [loss_normalcontrolnet_lpips.item()]
            else:
                loss_dict["loss_normalcontrolnet_lpips"].append(loss_normalcontrolnet_lpips.item())

            loss_normalcontrolnet = (loss_normalcontrolnet_L1 + loss_normalcontrolnet_lpips) * self.opt.lambda_normalcontrolnet if iter_idx >= self.opt.iters_init else 0 
            
            batch_vis_dict["controlnet_refined_images"] = controlnet_refined_images
            batch_vis_dict["controlnet_control_images"] = F.interpolate(controlnet_control_images.float(), (render_resolution, render_resolution), mode="bilinear", align_corners=False)


            if "normalcontrolnet" not in loss_dict:
                loss_dict["normalcontrolnet"] = [loss_normalcontrolnet.item()]
            else:
                loss_dict["normalcontrolnet"].append(loss_normalcontrolnet.item())

            loss = loss + loss_normalcontrolnet


        # save a copy of images and refined_images for visualization
        # detach the visualization and save to step_vis_dict
        for vis_mode in ["images, refined_images", "controlnet_refined_images", "controlnet_control_images"]:
            if vis_mode in batch_vis_dict and len(batch_vis_dict[vis_mode]) > 0:
                step_vis_dict[vis_mode].append(batch_vis_dict[vis_mode].detach()[0].unsqueeze(0))

        save_size = 256
        for vis_mode in ["rendered_albedos", "rendered_target_albedos",  "rendered_lambertians", "rendered_labels", "rendered_albedos_patch", "rendered_target_albedos_patch", "rendered_perturb_normals", "rendered_target_perturb_normals", "rendered_guidance_perturb_normals", "rendered_target_perturb_normal2s", "rendered_guidance_perturb_normal2s", "rendered_masks", "rendered_labels_patch", "rendered_masks_patch"]:
            if vis_mode in batch_vis_dict and len(batch_vis_dict[vis_mode]) > 0:
                if "label" in vis_mode:
                    step_vis_dict[vis_mode].append(F.interpolate(batch_vis_dict[vis_mode].detach()[0].unsqueeze(0), (save_size, save_size), mode="nearest"))
                else:
                    vis_data = (batch_vis_dict[vis_mode].detach()[0].unsqueeze(0) + 1.0) / 2 if "perturb_normal" in vis_mode else batch_vis_dict[vis_mode].detach()[0].unsqueeze(0)
                    step_vis_dict[vis_mode].append(F.interpolate(vis_data, (save_size, save_size), mode="bilinear", align_corners=False)) # align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear
  

        if self.opt.num_part_label > 0:
            # masks [2, H, W]
            step_vis_dict["seg_masks_partA"].append(F.interpolate(partA_mask.detach(), (save_size, save_size), mode="bilinear"))
            step_vis_dict["seg_masks_partB"].append(F.interpolate(partB_mask.detach(), (save_size, save_size), mode="bilinear"))
            step_vis_dict["seg_masks_partA_rendered"].append(F.interpolate(partA_mask_rendered.detach(), (save_size, save_size), mode="bilinear"))
            step_vis_dict["seg_masks_partB_rendered"].append(F.interpolate(partB_mask_rendered.detach(), (save_size, save_size), mode="bilinear"))
            step_vis_dict["seg_masks_partA_rendered_patch"].append(F.interpolate(partA_mask_patch_rendered.detach(), (save_size, save_size), mode="bilinear"))
            step_vis_dict["seg_masks_partB_rendered_patch"].append(F.interpolate(partB_mask_patch_rendered.detach(), (save_size, save_size), mode="bilinear"))

        
        # optimization
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
            
        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        output = {}
        
        # update the loss_dict for this iteration
        if return_loss_dict:
            for k, v in loss_dict.items():
                if isinstance(v, list):
                    loss_dict[k] = np.mean(v)
            output["loss_dict"] = loss_dict

        # save the images and refined_images for visualization
        for k, k_list in step_vis_dict.items():
            if "label" not in k:
                if len(k_list) > 0:
                    step_vis_dict[k] = torch.cat(k_list, dim=0).cpu().numpy()
                else:
                    step_vis_dict[k] = []


        output_rendered_labels = np.zeros_like(step_vis_dict["rendered_albedos"])            
        output_rendered_labels[:, :2] = torch.cat(step_vis_dict["rendered_labels"], dim=0).cpu().numpy()
        step_vis_dict["rendered_labels"] = output_rendered_labels

        output_rendered_labels_patch = np.zeros_like(step_vis_dict["rendered_albedos_patch"])
        output_rendered_labels_patch[:, :2] = torch.cat(step_vis_dict["rendered_labels_patch"], dim=0).cpu().numpy()
        step_vis_dict["rendered_labels_patch"] = output_rendered_labels_patch
        
        output["vis_dict"] = step_vis_dict

        return output

    
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(
            img, (self.W, self.H), interpolation=cv2.INTER_AREA
        )
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (
            1 - self.input_mask
        )
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()
    
    def save_model(self, postfix="", save_frame=False, num_frames=1):


        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.opt.save_path + postfix + '.' + self.opt.mesh_format)
        self.renderer.export_mesh(path)
        print(f"[INFO] save model to {path}.")

        # save current opt to a json file
        opt_dict = OmegaConf.to_container(opt, resolve=True)
        # export opt_dict to json
        opt_output_path = os.path.join(self.opt.outdir, self.opt.save_path + postfix + '_opt.json')
        with open(opt_output_path, 'w') as f:
            json.dump(opt_dict, f, indent=4)
        print(f"Save the current opt to {opt_output_path}")

        if hasattr(self, "loss_dict_all"):
            import pickle
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_loss_dict_all.pkl')
            print(f"[INFO] save loss_dict_all to {path}.")
            with open(path, 'wb') as f:
                pickle.dump(self.loss_dict_all, f)
            
            # plot the loss
            assert "iter" in self.loss_dict_all, "loss_dict_all should have 'iter' key!"
            import matplotlib.pyplot as plt
            # initialize the plot
            plt.figure()
            for k, v in self.loss_dict_all.items():
                if k == "iter":
                    continue
                else:
                    start_iter = self.loss_dict_all[k]["start_iter"]
                    stop = start_iter + len(self.loss_dict_all[k]["loss"])
                    plt.plot(np.arange(start_iter, stop), self.loss_dict_all[k]["loss"], label=k)
            plt.legend()
            plt.savefig(os.path.join(self.opt.outdir, self.opt.save_path + '_loss_plot.png'))
            plt.close()
        
        # save image visualizations as videos
        for k, k_list in self.vis_dict.items():
            if len(k_list) > 0 and "rendered_patch" not in k:
                video_output_path = os.path.join(self.opt.outdir, self.opt.save_path + f'_{k}_list.mp4')
                convert_images_to_video(k_list, video_output_path, fps=3, save_frame=save_frame, num_frames=num_frames)
                print(f"[INFO] save {k} to {video_output_path}")



        # create concatenated video for rendered_target_albedos_patch_list, seg_masks_partA_rendered_patch_list, seg_masks_partB_rendered_patch_list
        if self.vis_dict["seg_masks_partB_rendered_patch"] is not None and len(self.vis_dict["seg_masks_partB_rendered_patch"]) > 0:
            video_output_path = os.path.join(self.opt.outdir, self.opt.save_path + '_concat_patch_masks.mp4')

            seg_masks_partA_frames = [(np.array(v).squeeze(0).transpose(1, 2, 0).astype(np.float32)*255).astype(np.uint8) for k, v in self.vis_dict["seg_masks_partA_rendered_patch"].items()] # [N, 256, 256, 1]
            seg_masks_partA_frames = [np.concatenate([mask, mask, mask], axis=-1) for mask in seg_masks_partA_frames] # convert mask to 3 channels
            seg_masks_partB_frames = [(np.array(v).squeeze(0).transpose(1, 2, 0).astype(np.float32)*255).astype(np.uint8) for k, v in self.vis_dict["seg_masks_partB_rendered_patch"].items()]
            
            seg_masks_partB_frames = [np.concatenate([mask, mask, mask], axis=-1) for mask in seg_masks_partB_frames]
            rendered_albedos_frames = [(np.array(v).squeeze(0).transpose(1, 2, 0).astype(np.float32)*255).astype(np.uint8) for k, v in self.vis_dict["rendered_target_albedos_patch"].items()]
            
            concat_images_list = []
            for i in range(len(seg_masks_partA_frames)):
                concat_images = np.concatenate([seg_masks_partA_frames[i], seg_masks_partB_frames[i], rendered_albedos_frames[i]], axis=1)
                concat_images_list.append(concat_images)
            imageio.mimwrite(video_output_path, concat_images_list, fps=3, quality=8, macro_block_size=1)
            print(f"[INFO] save rendered_targt_albedos_patch_list, seg_masks_partA_rendered_patch_list, seg_masks_partB_rendered_patch_list to {video_output_path}")

        
        # create concatenated video for rendered_albedos_list, rendered_lambertians_list, controlnet_refined_images_list
        if self.vis_dict["rendered_albedos"] is not None and len(self.vis_dict["controlnet_refined_images"]) > 0:
            # save the rendered_albedos, rendered_lambertians, controlnet_refined_images to video for debugging the SDS loss
            video_output_path = os.path.join(self.opt.outdir, self.opt.save_path + '_SDS_concat_rendering.mp4')
            # since self.controlnet_refined_images_list is shorter, take the last few frames of rendered_albedos_list and rendered_lambertians_list to have the same length as controlnet_refined_images_list
            # print(f"check self.controlnet_refined_images_list: {type(self.controlnet_refined_images_list)}, ") # dict
            # extract frames 
            controlnet_refined_images_frames = [(np.array(v).squeeze(0).transpose(1, 2, 0).astype(np.float32)*255).astype(np.uint8) for k, v in self.vis_dict["controlnet_refined_images"].items()]
            rendered_albedos_frames = [(np.array(v).squeeze(0).transpose(1, 2, 0).astype(np.float32)*255).astype(np.uint8) for k, v in self.vis_dict["rendered_albedos"].items()][-len(controlnet_refined_images_frames):]
            rendered_lambertians_frames = [(np.array(v).squeeze(0).transpose(1, 2, 0).astype(np.float32)*255).astype(np.uint8) for k, v in self.vis_dict["rendered_lambertians"].items()][-len(controlnet_refined_images_frames):]
            
            # resize controlnet_refined_images_frames to 256
            controlnet_refined_images_frames = [cv2.resize(img, (256, 256)) for img in controlnet_refined_images_frames]

            # concate three lists per frame
            concat_images_list = []
            for i in range(len(controlnet_refined_images_frames)):
                concat_images = np.concatenate([rendered_albedos_frames[i], rendered_lambertians_frames[i], controlnet_refined_images_frames[i]], axis=1)
                concat_images_list.append(concat_images)
            # save the concatenated images to video
            imageio.mimwrite(video_output_path, concat_images_list, fps=3, quality=8, macro_block_size=1)
            print(f"[INFO] save rendered_albedos_list, rendered_lambertians_list, controlnet_refined_images_list to {video_output_path}")


    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            self.loss_dict_all = {}
            self.vis_dict = {k: {} for k in self.vis_modes}

            # record current timestamp
            start_time = time.time()

            for i in tqdm.trange(iters):
                output = self.train_step(return_loss_dict=True, iter_idx=i)
                loss_dict = output["loss_dict"]
                
                for k, k_list in output["vis_dict"].items():
                    if len(k_list) > 0:
                        self.vis_dict[k][i] = k_list

                # update loss_dict_all
                for k, v in loss_dict.items():
                    if k not in self.loss_dict_all:
                        self.loss_dict_all[k] = {"start_iter": i, "loss": [v]}
                    else:
                        self.loss_dict_all[k]["loss"].append(v)
                if i == 0:
                    self.loss_dict_all["iter"] = [i]
                else:
                    self.loss_dict_all["iter"].append(i)

                if i == self.opt.iters_init - 1:
                    time_init = time.time() - start_time
                    print(f"[INFO] finish initialization training i {i} {type(i)}, iters_init {self.opt.iters_init}, taking {time_init}s. Start co-optimizing with rendering loss ...")

                    # save the albedo and normal map after initialization
                    self.save_model(postfix="_initialized", save_frame=self.opt.save_frame, num_frames=20)
                    start_time = time.time()

            time_coop = time.time() - start_time
            print(f"[INFO] finish co-optimization, taking {time_coop}s.")

        # save
        self.save_model(save_frame=self.opt.save_frame, num_frames=20)


if __name__ == "__main__":

    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # merge the config file with the command line arguments
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    opt.outdir = os.path.join(opt.outdir, opt.save_path)
    os.makedirs(opt.outdir, exist_ok=True)

    # edit the prompt based on the texture description. structured prompts
    mesh_obj = opt.mesh.split("/")[-2]
    texture_name = opt.tactile_texture_object
    if opt.num_part_label > 0:
        # multi-part texture generation
        opt.prompt, opt.negative_prompt, opt.partA_idx, opt.partB_idx = generate_textured_prompt(mesh_obj=mesh_obj, texture_name=texture_name, positive_prompt=None, negative_prompt=None, add_texture=True, multi_parts=True, texture2_name=opt.texture2_name)
    else:
        # single-part texture generation
        opt.prompt, opt.negative_prompt = generate_textured_prompt(mesh_obj=mesh_obj, texture_name=texture_name, positive_prompt=None, negative_prompt=None, add_texture=True)
    print(f"Set textured prompt: \nprompt: {opt.prompt} \nnegative_prompt: {opt.negative_prompt}")

    # override the tactile loss weights if no_tactile is set to True
    if opt.no_tactile or opt.no_train_tactile:
        opt.load_tactile = False
        opt.lambda_tactile_regularization = 0
        opt.lambda_tactile_regularization_init = 0
        opt.lambda_tactile_guidance = 0
    else:
        opt.load_tactile = True
        # parse the tacitle texture path
        opt.tactile_normal_path = os.path.join("./data/tactile_textures", f"{opt.tactile_texture_object}_tactile_texture_map_2_normal.png") 
        print(f"Tactile texture path: {opt.tactile_normal_path}")
        # add the second tactile texture path for multi-parts
        if opt.num_part_label > 0:
            assert opt.num_part_label == 2, f"Unsupported number of part label {opt.num_part_label}"
            opt.tactile_normal_path2 = os.path.join("./data/tactile_textures", f"{opt.texture2_name}_tactile_texture_map_2_normal.png")
            print(f"Tactile texture path 2: {opt.tactile_normal_path2}")
    
    # auto find mesh from stage 1
    if opt.mesh is None:
        default_path = os.path.join(opt.outdir, opt.save_path + '_mesh.' + opt.mesh_format)
        if os.path.exists(default_path):
            opt.mesh = default_path
        else:
            raise ValueError(f"Cannot find mesh from {default_path}, must specify --mesh explicitly!")

    gui = GUI(opt)
    gui.train(opt.iters_refine)

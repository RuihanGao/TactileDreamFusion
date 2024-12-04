from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from controlnet_aux import NormalBaeDetector
import torch
import torch.nn as nn
import torch.nn.functional as F
from guidance.cross_attention import prep_unet

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def convert_normal2bae(normal_image):
    """
    Convert a normal map from OpenGL convention (red - right, green - up, blue - front), range [-1, 1]
    to BAE convention (red - left, green - up, blue - front), range [0, 1]
    
    Args:
        normal_image: tensor, shape (N, 3, H, W), in OpenGL convention. (red - right, green - up, blue - front), range [-1, 1]
        
    Returns:
        normal_image: tensor, shape (N, 3, H, W), in BAE convention. (red - left, green - up, blue - front), range [0, 1]
    """

    # current background color is [0, 0, 0]
    # create a mask where the tensor value equals to the background color
    bg_mask = torch.all(normal_image == 0, dim=1).unsqueeze(1) # shape (N, 1, H, W)
    
    # set background color to [0, 0, 1]
    # create a tensor of shape (N, 3, H, W) with the background color [0, 0, 1]
    bg_tensor = torch.zeros_like(normal_image)
    bg_tensor[:, 2, :, :].fill_(1)
    # set the background color to the normal image given the mask
    normal_image = torch.where(bg_mask, bg_tensor, normal_image)

    # flip the red channel
    normal_image[:, 0, :, :] *= -1
    # normalize to unit length
    normal_image = F.normalize(normal_image, p=2, dim=1)
    # convert to range [0, 1]
    normal_image = (normal_image + 1.0) / 2.0
    return normal_image


class ControlNet(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="1.5",
        controlnet_name="lllyasviel/control_v11p_sd15_normalbae",
        t_range=[0.02, 0.1],
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        assert self.sd_version == "1.5", "Only stable diffusion v1.5 is supported for ControlNet."
        model_key = "runwayml/stable-diffusion-v1-5"

        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        # use prep_unet to save attention maps
        self.unet = prep_unet(pipe.unet)
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype
        )
        
        del pipe

        # Load Normal-conditioned-ControlNet
        self.controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=self.dtype).to(device)
        self.processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")

        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, torch_dtype=torch.float16, safety_checker=None, 
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)

        # Initialize attention maps (placeholder)
        self.cross_attn = {}
        self.self_attn = {}


    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds
    
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def refine(self, pred_rgb, control_images=None,
               guidance_scale=100, steps=50, strength=0.8, controlnet_conditioning_scale=1.0, attn_timestep=181, return_attn=False
        ):

        """
        Generate a reference image given (the prompt, control normal image, and current rgb rendering)
        (Assume the prompt has been converted to text embeddings and stored in self.embeddings)
        Args:
            prompt: str
            control_images: normal map, tensor, shape (N, 3, H, W), in OpenGL convention. (red - right, green - up, blue - front), range [-1, 1]

        Returns:
            output_images: refined images, tensor, shape (N, 3, H, W), range [0, 1], dtype torch.float16
            attn_timestep: time step to extract attention maps (default 181)
        """

        if return_attn:
            # Initialize attention maps
            cross_attn = {}
            self_attn = {}
        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))

        if control_images is not None:
            # ControlNet takes a normal map in BAE convention (red - left, green - up, blue - front), range [0, 1]
            control_images = convert_normal2bae(control_images)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            
            latent_model_input = torch.cat([latents] * 2)

            if control_images is not None:
                # save the original copy of control_images for logging purposes
                control_images_embed = control_images.clone().detach().to(self.dtype).to(self.device)
                control_images_embed = torch.cat([control_images_embed] * 2) # [2, 3, 512, 512]

                # forward pass of controlnet
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input.to(torch.float16),
                    t.repeat(latent_model_input.shape[0]).to(self.controlnet.device),
                    encoder_hidden_states=embeddings,
                    controlnet_cond=control_images_embed,
                    return_dict=False,
                )

                down_block_res_samples = [
                    down_block_res_sample * controlnet_conditioning_scale
                    for down_block_res_sample in down_block_res_samples
                ]
                mid_block_res_sample *= controlnet_conditioning_scale
              
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input.to(torch.float16),
                    t,
                    encoder_hidden_states=embeddings,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            
            else:
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=embeddings,
                ).sample

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if return_attn:
                # add the cross attention map to the dictionary
                if t.item() == attn_timestep:
                    cross_attn[t.item()] = {}
                    self_attn[t.item()] = {}
                    for name, module in self.unet.named_modules():
                        module_name = type(module).__name__
                        if "Cross" in module_name or "cross" in module_name:
                            for i, layer in enumerate(module.attentions):
                                attn_name = f"{name}_{i}"
                                # offload to avoid OOM
                                cross_attn[t.item()][attn_name] = layer.transformer_blocks[0].attn2.attn_probs.detach()
                                self_attn[t.item()][attn_name] = layer.transformer_blocks[0].attn1.attn_probs.detach()

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        if return_attn:
            self_attn_list = [value for key,value in self_attn[attn_timestep].items()]
            cross_attn_list = [value for key,value in cross_attn[attn_timestep].items()]
            return self_attn_list, cross_attn_list
        else:
            return imgs, control_images


    def train_step(
        self,
        pred_rgb,
        control_images,
        guidance_scale=100,
        timestep_t=None,
        as_latent=False,
        controlnet_conditioning_scale=1.0,
    ): 
        """
        Given a predicted image, compute the single-step SDS loss with normal-conditioned ControlNet.
        (Assume the prompt has been converted to text embeddings and stored in self.embeddings)
        Args:
            pred_rgb: tensor, shape (N, 3, H, W), range [0, 1], dtype torch.float32
            timestep_t: float, range [0, 1], the ratio of the step to take in the diffusion process
            guidance_scale: float, the scale of the guidance
            as_latent: bool, whether the input pred_rgb is in latent space

        """
        torch.cuda.synchronize()
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            latents = self.encode_imgs(pred_rgb_512) # [1, 4, 64, 64]
        
        if control_images is not None:
            # ControlNet takes a normal map in BAE convention (red - left, green - up, blue - front), range [0, 1]
            control_images = convert_normal2bae(control_images)

        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
        
        # sample timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if timestep_t is not None:
            t = int(self.min_step + ((self.max_step - self.min_step) * timestep_t))
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        tt = torch.cat([t] * 2).to(self.controlnet.device) # [2]

        # predict the noise residual with unet. NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)

            if control_images is not None:
                # save the original copy of control_images for logging purposes
                control_images_embed = control_images.clone().detach().to(self.dtype).to(self.device)
                control_images_embed = torch.cat([control_images_embed] * 2)
                
                # forward pass of controlnet
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input.to(torch.float16),
                    tt.to(self.controlnet.device),
                    encoder_hidden_states=embeddings,
                    controlnet_cond=control_images_embed,
                    return_dict=False,
                )

                down_block_res_samples = [
                    down_block_res_sample * controlnet_conditioning_scale
                    for down_block_res_sample in down_block_res_samples
                ]
                mid_block_res_sample *= controlnet_conditioning_scale

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input.to(torch.float16),
                    tt,
                    encoder_hidden_states=embeddings,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

            else:            
                noise_pred = self.unet(latent_model_input.to(torch.float16), tt, encoder_hidden_states=embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        # w(t), sigma_t^2
        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w[:,None,None,None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='mean') / latents.shape[0]

        return loss, control_images, target


    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    1,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings
            ).sample

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):  
        latents = 1 / self.vae.config.scaling_factor * latents.to(self.dtype)
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


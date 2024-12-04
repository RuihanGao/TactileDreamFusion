import numpy as np
import torch
import imageio
import cv2
import pdb

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def cross(x, y):
    if isinstance(x, np.ndarray):
        return np.cross(x, y)
    else:
        return torch.cross(x, y)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def create_uv_coords(texture_size, device="cpu"):
    grid_u, grid_v = torch.meshgrid(torch.linspace(0, 1, texture_size), torch.linspace(0, 1, texture_size)) # shape [texture_size, texture_size]
    uv = torch.stack((grid_u, grid_v), dim=-1).reshape(-1, 2) # shape [texture_size, texture_size, 2] -> [texture_size*texture_size, 2]
    return uv.to(device)



def convert_images_to_video(images_json, output_path, fps=3, save_frame=False, num_frames=1):

    # images = [(np.array(v).squeeze(0).transpose(1, 2, 0).astype(np.float32)*255).astype(np.uint8) for k, v in images_json.items()]
    images = [(np.array(v)[0].transpose(1, 2, 0).astype(np.float32)*255).astype(np.uint8) for k, v in images_json.items()]

    indexes = [k for k, v in images_json.items()]

    # Video properties
    output_images = []

    # Add text to each image and write to video
    for image, index in zip(images, indexes):
        # Convert the image to BGR format
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output_images.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    imageio.mimwrite(output_path, images, fps=fps, quality=8, macro_block_size=1)
    
    if save_frame:
        # save the last few frames
        num_frames = min(num_frames, len(output_images))
        for frame_idx in range(1, num_frames):
            frame_output_path = output_path.replace(".mp4", f"_-{frame_idx}.png")
            imageio.imwrite(frame_output_path, output_images[-1*frame_idx])

### Visualization functions ###
# swipe screen function, per frame
def lin_brush(col_g, color, t):
    t = np.sin(t*np.pi)
    imgw = col_g.shape[1]
    wpx = int(t*imgw)
    img_new = np.zeros_like(color)
    img_new[:,:wpx] = color[:,:wpx]
    img_new[:,wpx:] = col_g[:,wpx:]

    return img_new


def generate_textured_prompt(mesh_obj, texture_name, positive_prompt=None, negative_prompt=None, add_texture=True, add_positive_prompt=True, multi_parts=False, texture2_name=None):
    """
    Given a mesh object and a tactile texture name, query the pre-defined look-up table for the texture description, and complete the prompt.

    Args:
        multi_parts: bool, option to use multi-parts prompt
    """
    import os
    import json

    default_positive_prompt = ", highly detailed, hd, best quality, front lighting"
    default_negative_prompt = "bad quality, blurred, low resolution, low quality, worst quality, low res, glitch, deformed, mutated, ugly, disfigured"
    # set default prompts
    positive_prompt = default_positive_prompt if positive_prompt is None else positive_prompt
    negative_prompt = default_negative_prompt if negative_prompt is None else negative_prompt

    # edit the prompt based on the texture description. structured prompts
    texture_description_path = "./texture_desc.json"
    with open(texture_description_path, "r") as f:
        texture_desc_dict = json.load(f)
    assert texture_name in texture_desc_dict.keys(), f"Cannot find texture description for {texture_name}"
    texture_desc = texture_desc_dict[texture_name]

    mesh_obj_dict = {
        "a_table_tennis_racket": {"desc": "a racket with red face and wooden handle",
                                  "partA_idx": 5, "partB_idx": 8},
        "a_gold_goat_sculpture": {"desc": "a goat sculpture with gold goat and black base",
                                    "partA_idx": 6, "partB_idx": 9},
        "a_potted_cactus": {"desc": "a cactus in a pot",
                            "partA_idx": 2, "partB_idx": 5},
        "lamp1": {"desc": "a lamp with white lampshade and wooden base",
                "partA_idx": 5, "partB_idx": 8},
        "lamp3": {"desc": "a lamp with beige lampshade and wooden base",
                "partA_idx": 5, "partB_idx": 8},
        "a_cactus_in_a_pot_2": {"desc": "a cactus in a pot",
                    "partA_idx": 2, "partB_idx": 5},
        "a_cactus_in_a_pot_3": {"desc": "a cactus in a pot",
                    "partA_idx": 2, "partB_idx": 5},
    }

    if multi_parts:
        assert texture2_name is not None, "Please provide the second texture name for multi-parts prompt"
        assert texture2_name in texture_desc_dict.keys(), f"Cannot find texture description for {texture2_name}"
        texture_desc2 = texture_desc_dict[texture2_name]
        multi_parts_dict = {
            "a_table_tennis_racket":{"partA": "face", "partB": "handle"},
            "a_gold_goat_sculpture": {"partA": "goat", "partB": "base"},
            "a_potted_cactus": {"partA": "cactus", "partB": "pot"},
            "lamp1": {"partA": "lampshade", "partB": "base"},
            "lamp3": {"partA": "lampshade", "partB": "base"},
            "a_cactus_in_a_pot_2": {"partA": "cactus", "partB": "pot"},
            "a_cactus_in_a_pot_3": {"partA": "cactus", "partB": "pot"},
            
        }
        assert mesh_obj in multi_parts_dict.keys(), f"Cannot find multi-parts description for {mesh_obj}"
        assert mesh_obj in mesh_obj_dict.keys(), f"Cannot find mesh object description for {mesh_obj}"
        partA = multi_parts_dict[mesh_obj]["partA"]
        partB = multi_parts_dict[mesh_obj]["partB"]

    # mesh_obj_desc_dict = {"a_strawberry": "a strawberry, no leaves, no stem ", "an_avocado": "an avocado"}

    if not add_positive_prompt:
        positive_prompt = ""
    if add_texture:
        if not multi_parts:
            # single part
            # if mesh_obj contains numbers, remove it
            mesh_obj_name = [i for i in mesh_obj.split('_') if not i.isdigit()]
            # remove certain string from the list
            mesh_obj_name = [i for i in mesh_obj_name if i not in ["RichDreamer", "InstantMesh"]]
            # join the list to form a string
            mesh_obj_name = ' '.join(mesh_obj_name)
            prompt = f"{mesh_obj_name} with {texture_desc} texture" + positive_prompt
        else:
            mesh_desc = mesh_obj_dict[mesh_obj]["desc"]
            prompt = f"{mesh_desc}, {partA} with {texture_desc} texture, {partB} with {texture_desc2} texture" + positive_prompt
            partA_idx = mesh_obj_dict[mesh_obj]["partA_idx"]; partB_idx = mesh_obj_dict[mesh_obj]["partB_idx"]
            # compute the index of string "partA" in the prompt. index starts from 1
            # partA_idx = len(mesh_desc.split(' ')) + 1
            # partB_idx = len(mesh_desc.split(' ')) + len(partA.split(' ')) + 1 + len(texture_desc.split(' ')) + 1 + 1
            print(f"In prompt {prompt}, \n partA {partA} is at index {partA_idx}, \n partB {partB} is at index {partB_idx}")
    else:
        prompt = f"{mesh_obj.replace('_', ' ')}" + positive_prompt
        # prompt = f"{mesh_obj_desc_dict[mesh_obj]}" + positive_prompt

    if multi_parts:
        return prompt, negative_prompt, partA_idx, partB_idx
    return prompt, negative_prompt


import random

def toggle_variable(probability):
    if random.random() < probability:
        return 1
    else:
        return 0

# differentiation
def generate_gradients(height_map):
    """
    Generate the gradients of the height map.
    :param height_map: np.2darray (H, W); the height map.
    :return gxy: np.3darray (H, W, 2); the gradients.
    """

    [h, w] = height_map.shape
    center = height_map[1 : h - 1, 1 : w - 1]  # z(x,y)
    top = height_map[0 : h - 2, 1 : w - 1]  # z(x,y-1)
    bot = height_map[2:h, 1 : w - 1]  # z(x,y+1)
    left = height_map[1 : h - 1, 0 : w - 2]  # z(x-1,y)
    right = height_map[1 : h - 1, 2:w]  # z(x+1,y)
    # The direction here follows OpenGL convention, x (red) -- right, y (green) -- up, z (blue) -- out of the screen.
    dzdy = (top - bot) / 2.0
    dzdx = (right - left) / 2.0

    def padding(x):
        """
        Padding the input array by duplicating the edge value.
        :param x: np.2darray (H, W); the input array.
        :return x_pad: np.2darray (H+2, W+2); the padded array.
        """
        x_pad = np.pad(x, ((1, 1), (1, 1)), 'edge')
        return x_pad

    gx = padding(dzdx)
    gy = padding(dzdy)
    gxy = np.stack([gx, gy], axis=-1)
    return gxy
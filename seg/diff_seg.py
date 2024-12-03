'''
Modified from https://github.com/google/diffseg/blob/main/diffseg/segmentor.py
'''

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

def aggregate_x_weights(weight_list, weight_ratio=None, device="cpu"):
    # x_weights: 8 x size**2 x 77
    # return 512 x 512 x 77
    # if weight_ratio is None:
    #   weight_ratio = self.get_weight_ratio(weight_list)
    aggre_weights = torch.zeros((512, 512, 77), dtype=torch.float32).to(device)

    for index, weights in enumerate(weight_list):
      size = int(torch.sqrt(torch.tensor(weights.shape[-2])).to(device))
      ratio = 512 // size
      weights = weights.mean(0).reshape(1, size, size, -1).permute(0, 3, 1, 2)  # Change to (N, C, H, W) format

      # Ensure the weights are in float32 before upsampling
      if weights.dtype != torch.float32:
          weights = weights.type(torch.float32)
          
      weights = F.interpolate(weights, scale_factor=ratio, mode='bilinear', align_corners=False, recompute_scale_factor=True)
      sum_weights = weights.sum(dim=1, keepdim=True)
      weights = weights / sum_weights
      aggre_weights += weights.permute(0, 2, 3, 1).squeeze(0) * weight_ratio[index]

    return aggre_weights

def get_weight_ratio(weight_list):
    # This function assigns proportional aggergation weight 
    sizes = []
    for weights in weight_list:
      sizes.append(np.sqrt(weights.shape[-2]))
    denom = np.sum(sizes)
    return sizes / denom

def aggregate_weights(weight_list, weight_ratio=None, device="cpu"):
    """
    Return:
    aggre_weights: tensor. float32
    """
    if weight_ratio is None:
        weight_ratio = get_weight_ratio(weight_list)
    aggre_weights = torch.zeros((64, 64, 64, 64), dtype=torch.float32).to(device)
    
    for index, weights in enumerate(weight_list):
        size = int(torch.sqrt(torch.tensor(weights.shape[-1])).to(device))
        ratio = 64 // size
        # Average over the multi-head channel
        weights = weights.mean(0).reshape(-1, size, size)

        # Ensure the weights are in float32 before upsampling
        if weights.dtype != torch.float32:
            weights = weights.type(torch.float32)

        # Upsample the last two dimensions to 64 x 64
        weights = weights.unsqueeze(1)  # Add channel dimension
        # print(weights.shape)
        weights = F.interpolate(weights, size=(64, 64), mode='bilinear', align_corners=False)
        # print(weights.shape)
        weights = weights.reshape(size, size, 64, 64)

        # Normalize to make sure each map sums to one
        weights = weights / weights.sum(dim=(2, 3), keepdim=True)
        
        # Spatial tiling along the first two dimensions
        weights = weights.repeat_interleave(ratio, dim=0)
        weights = weights.repeat_interleave(ratio, dim=1)

        # Aggregate according to weight_ratio
        aggre_weights += weights * weight_ratio[index]
    
    return aggre_weights.to(torch.float32)

def KL(X, Y):
    quotient = torch.log(X) - torch.log(Y)
    kl_1 = torch.sum(X * quotient, dim=(-2, -1)) / 2
    kl_2 = -torch.sum(Y * quotient, dim=(-2, -1)) / 2
    return kl_1 + kl_2

def mask_merge(iter, attns, kl_threshold, grid=None):
    attns = torch.tensor(attns, dtype=torch.float32)  # Ensure dtype for PyTorch operations
    if iter == 0:
        # The first iteration of merging
        anchors = attns[grid[:, 0], grid[:, 1], :, :]  # 256 x 64 x 64
        anchors = anchors.unsqueeze(1)  # 256 x 1 x 64 x 64
        attns = attns.reshape(1, 4096, 64, 64)
        # Splitting into portions
        split = int(np.sqrt(grid.shape[0]))
        kl_bin = []
        for i in range(split):
            temp = KL(anchors[i * split:(i + 1) * split].half(), attns.half()) < kl_threshold[iter]
            kl_bin.append(temp)
        kl_bin = torch.cat(kl_bin, axis=0).to(torch.float32)  # 256 x 4096
        new_attns = torch.matmul(kl_bin, attns.reshape(-1, 4096).T) / torch.sum(kl_bin, dim=1, keepdim=True)
        new_attns = new_attns.reshape(-1, 64, 64)  # 256 x 64 x 64
    else:
        # The rest of merging iterations, reducing the number of masks
        matched = set()
        new_attns = []
        for i,point in enumerate(attns):
            if i in matched:
                continue
            matched.add(i)
            anchor = point
            kl_bin = (KL(anchor,attns) < kl_threshold[iter]).cpu().numpy() # 64 x 64
            if kl_bin.sum() > 0:
                matched_idx = np.arange(len(attns))[kl_bin.reshape(-1)]
            for idx in matched_idx: matched.add(idx)
            aggregated_attn = attns[kl_bin].mean(0)
            new_attns.append(aggregated_attn.reshape(1,64,64))
        new_attns = torch.cat(new_attns)

    return new_attns

def generate_sampling_grid(num_of_points):
    segment_len = 63//(num_of_points-1)
    total_len = segment_len*(num_of_points-1)
    start_point = (63 - total_len)//2
    x_new = np.linspace(start_point, total_len+start_point, num_of_points)
    y_new = np.linspace(start_point, total_len+start_point, num_of_points)
    x_new,y_new=np.meshgrid(x_new,y_new,indexing='ij')
    points = np.concatenate(([x_new.reshape(-1,1),y_new.reshape(-1,1)]),axis=-1).astype(int)
    return points

def calculate_dist(self_attn_merged, cross_attn):
    '''
    self_attn_merged: K x size x size
    cross_attn: N x size x size
    '''
    self_attn_merged = self_attn_merged.reshape(-1, 512*512)
    cross_attn = cross_attn.reshape(-1, 512*512)

    # Normalize self_attn_merged and cross_attn
    self_attn_merged = self_attn_merged / torch.norm(self_attn_merged, dim=1, keepdim=True, p=1)
    cross_attn = cross_attn / torch.norm(cross_attn, dim=1, keepdim=True, p=1)

    # dist = torch.cdist(self_attn_merged, cross_attn)
    dist = torch.zeros((cross_attn.shape[0], self_attn_merged.shape[0]))
    for i in range(cross_attn.shape[0]):
        for j in range(self_attn_merged.shape[0]):
            dist[i,j] = KL(cross_attn[i].reshape(512,512), self_attn_merged[j].reshape(512,512))

    return dist

def get_semantics(pred, x_weight, voting="majority"):
    # This function assigns semantic labels to masks 
    
    # x_weight, size x size x N
    x_weight = x_weight.reshape(512 * 512, -1)
    
    # Normalize the cross-attention maps spatially using PyTorch
    norm = torch.norm(x_weight, dim=0, keepdim=True, p=1)
    x_weight = x_weight / norm
    
    pred = pred.reshape(512 * 512, -1)

    label_to_mask = defaultdict(list)
    for i in set(pred.flatten().tolist()):  # Convert tensor to list to create a set
        mask = (pred == i).flatten()
        if voting == "majority":
            logits = x_weight[mask, :]
            index = logits.argmax(dim=-1)
            category = int(torch.mode(index)[0].item())  # Get median and convert to int
        else:
            logit = x_weight[mask, :].mean(dim=0)
            category = logit.argmax(dim=-1).item() # Get argmax and convert to item
        label_to_mask[category].append(i)
    return label_to_mask

def seg_attn(self_attn_list, cross_attn_list, token_indices, fg_mask=None):
    """
    Args:
    fg_mask: [H, W]
    """
    cross_attn_aggre = aggregate_x_weights(weight_list=cross_attn_list, weight_ratio=[1.0 for x in cross_attn_list], device=cross_attn_list[0].device) # 512x512x77
    self_attn_aggre = aggregate_weights(weight_list=self_attn_list, device=self_attn_list[0].device)

    grid = generate_sampling_grid(16)

    attns_merged = mask_merge(0, self_attn_aggre, kl_threshold=[0.9]*3, grid=grid)

    attns_merged = mask_merge(1, attns_merged, kl_threshold=[0.9]*3)
    attns_merged = mask_merge(2, attns_merged, kl_threshold=[0.9]*3)

    # Upsampling
    attns_merged = F.interpolate(attns_merged.unsqueeze(1), scale_factor=(8, 8), mode='bilinear', align_corners=False).squeeze(1) # K x 512 x 512

    # Non-Maximum Suppression
    label_maps = torch.argmax(attns_merged, dim=0).reshape(512, 512)

    index_to_labels = get_semantics(label_maps, cross_attn_aggre[..., token_indices], voting='mean')

    masks = []
    # for key, labels in index_to_labels.items():
    for key in range(len(token_indices)):
        labels = index_to_labels[key]
        mask = torch.zeros_like(label_maps)
        for label in labels:
            mask = mask + (label_maps == label)
        if fg_mask is not None:
            mask = mask * fg_mask
        masks.append(mask)

    masks = torch.stack(masks, dim=0)
    masks = masks > 0
    return masks


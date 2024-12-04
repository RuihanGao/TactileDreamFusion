import torch
# diffusers rename CrossAttention to Attention. Ref: https://github.com/huggingface/diffusers/issues/4969#issuecomment-1713590616
from diffusers.models.attention import Attention as CrossAttention

"""
Modify the attention processor in Stable Diffusion - unet to save the attention probabilities.
Ref: pix2pix-zero https://github.com/pix2pixzero/pix2pix-zero/blob/main/src/utils/cross_attention.py
"""

class MyCrossAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # new bookkeeping to save the attn probs
        attn.attn_probs = attention_probs

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def prep_unet(unet):
    """
    Set the original forward pass to be performed by a custom cross attention processor. 
    We only extract the attention probabilities from the attention module, not training the model.

    Parameters:
    unet: A U-Net model.

    Returns:
    unet: The prepared U-Net model.
    """

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention":
            module.set_processor(MyCrossAttnProcessor())
    return unet

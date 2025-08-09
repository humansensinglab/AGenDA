from pathlib import Path
from typing import List, Type, Any, Dict, Tuple, Union
import math

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


class UNetCrossAttentionHooker():
    def __init__(
            self,
            is_train: bool=True,
            latent_hw: int=64
    ):
        self.cross_attn_maps = []
        self.is_train=is_train
        self.latent_hw = latent_hw

    
    def clear(self):
        self.cross_attn_maps.clear()

    def _unravel_attn(self, x, n_heads):
        # type: (torch.Tensor, int) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        # n_heads: int
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        maps = []
        h = w = int(math.sqrt(x.size(1)))
        x = x.permute(2, 0, 1)

        for map_ in x:
            map_ = map_.view(map_.size(0), h, w)
            if not self.is_train:
                map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
            maps.append(map_)

        maps = torch.stack(maps, 0)  # (tokens, heads, height, width)
        maps=maps.permute(1, 0, 2 ,3) # (batch_size*heads, tokens, height, width)
        maps=maps.reshape([maps.shape[0]//n_heads, n_heads, *maps.shape[1:]]) # (batch_size, heads, tokens, height, width)
        maps = maps.mean(dim=1)
        return maps
    

    def compute_global_heat_map(self):
        """
        Compute the global heat map for each latent pixel, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).
        Returns:
            A heat map object for computing latent pixel-level heat maps.
        """
        heat_maps = self.cross_attn_maps

        all_merges = []

        with autocast(dtype=torch.float32):
            for heat_map in heat_maps:
                all_merges.append(F.interpolate(heat_map, size=(self.latent_hw, self.latent_hw), mode='bicubic').clamp_(min=0))

            try:
                maps = torch.stack(all_merges, axis=0)
            except RuntimeError:
                raise RuntimeError('No heat maps found.')

            maps = torch.mean(maps, axis=0)

        return maps

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
    ):
        """Capture attentions and aggregate them."""
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attn=(encoder_hidden_states is not None)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if is_cross_attn:
            maps = self._unravel_attn(attention_probs, attn.heads)
            self.cross_attn_maps.append(maps)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
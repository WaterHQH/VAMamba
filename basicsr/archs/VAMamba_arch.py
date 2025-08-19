# Code Implementation of the VAMamba Model
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from basicsr.utils.registry import ARCH_REGISTRY
from timm.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import collections


# Visualization
import os
import matplotlib.pyplot as plt
import numpy as np




NEG_INF = -1000000

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Layer
    Used to reduce dimension and add learnable low-rank adaptation
    """
    def __init__(self, in_dim, out_dim, rank=16, alpha=1.0, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA's A and B matrices
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # LoRA: x @ A @ B
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return self.dropout(lora_output)


class FeatureCache:
    """
    Feature cache mechanism
    Used to store and reuse historical features
    """
    def __init__(self, max_size=10, cache_dim=None):
        self.max_size = max_size
        self.cache_dim = cache_dim
        self.cache = collections.deque(maxlen=max_size)
        self.cache_keys = collections.deque(maxlen=max_size)
    
#     def update(self, features, key=None):
#         """Update cache"""
#         if key is None:
#             key = len(self.cache)
        
#         # If cache is empty, record feature dimension
#         if self.cache_dim is None:
#             self.cache_dim = features.shape[-1]
        
#         # Store features and corresponding key
#         self.cache.append(features.detach())
#         self.cache_keys.append(key)
    
#     def get_cached_features(self, current_features, similarity_threshold=0.8):
#         """Get similar historical features"""
#         if len(self.cache) == 0:
#             return None,0.0
        
#         # Calculate similarity with current features
#         similarities = []
#         for cached_feat in self.cache:
#             # Align spatial dimensions
#             if cached_feat.shape[1:3] != current_features.shape[1:3]:
#                 # [B, H, W, C] -> [B, C, H, W]
#                 cached_feat_ = cached_feat.permute(0, 3, 1, 2)
#                 # Interpolate to current feature's H, W
#                 cached_feat_ = F.interpolate(cached_feat_, size=current_features.shape[1:3], mode='bilinear')
#                 # [B, C, H, W] -> [B, H, W, C]
#                 cached_feat_ = cached_feat_.permute(0, 2, 3, 1)
#             else:
#                 cached_feat_ = cached_feat

#             sim = F.cosine_similarity(
#                 current_features.flatten(1),
#                 cached_feat_.flatten(1),
#                 dim=1
#             ).mean()
#             similarities.append(sim.item())
        
#         # Find the most similar feature
#         max_sim_idx = similarities.index(max(similarities))
#         max_similarity = similarities[max_sim_idx]
        
#         if max_similarity > similarity_threshold:
#             return self.cache[max_sim_idx], max_similarity
#         return None, 0.0
    def update(self, features, key=None):
        # features: [B, H, W, C]
        B = features.shape[0]
        for b in range(B):
            single_feat = features[b:b+1]  # Keep batch dimension as 1
            single_key = f"{key}_{b}" if key is not None else len(self.cache)
            if self.cache_dim is None:
                self.cache_dim = single_feat.shape[-1]
            self.cache.append(single_feat.detach())
            self.cache_keys.append(single_key)

    def get_cached_features(self, current_features, similarity_threshold=0.8):
        assert current_features is not None, "current_features is None"
        assert current_features.shape[0] > 0, "Input is empty, batch_size=0"
        B = current_features.shape[0]
        results = []
        for b in range(B):
            candidates = []
            for cached_feat in self.cache:
                cf = cached_feat[0]
                candidates.append(cf)
            cur = current_features[b]
            assert torch.is_tensor(cur), f"current_features[{b}] is not tensor, but {type(cur)}"
            similarities = []
            for cf in candidates:
                if cf.shape[:2] != cur.shape[:2]:
                    cf_ = cf.permute(2, 0, 1).unsqueeze(0)
                    cf_ = F.interpolate(cf_, size=cur.shape[:2], mode='bilinear')
                    cf_ = cf_.squeeze(0).permute(1, 2, 0)
                else:
                    cf_ = cf
                sim = F.cosine_similarity(
                    cur.flatten(0, 1), cf_.flatten(0, 1), dim=0
                ).mean()
                similarities.append(sim.item())
            if similarities and max(similarities) > similarity_threshold:
                max_sim_idx = similarities.index(max(similarities))
                max_similarity = similarities[max_sim_idx]
                best_feat = candidates[max_sim_idx]
                if best_feat.dim() == 3:
                    best_feat = best_feat.unsqueeze(0)
                results.append((best_feat, max_similarity))
            else:
                # No suitable cache, return None
                results.append((None, 0.0))
        return results

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.cache_keys.clear()
    
    def __len__(self):
        return len(self.cache)


class LoRACacheModule(nn.Module):
    def __init__(self, d_inner, lora_rank=16, cache_size=5, use_lora=True, use_cache=True):
        super().__init__()
        self.d_inner = d_inner
        self.lora_rank = lora_rank
        self.use_lora = use_lora
        self.use_cache = use_cache

        # LoRA layers
        if use_lora:
            self.lora_down = LoRALayer(d_inner, lora_rank, rank=lora_rank//2)
            self.lora_up = LoRALayer(lora_rank, d_inner, rank=lora_rank//2)

        
        if use_lora:
            self.feature_fusion = nn.Sequential(
                nn.Linear(d_inner + lora_rank, d_inner),
                nn.SiLU(),
                nn.Linear(d_inner, d_inner)
            )
        else:
            self.feature_fusion = nn.Sequential(
                nn.Linear(d_inner * 2, d_inner),
                nn.SiLU(),
                nn.Linear(d_inner, d_inner)
            )
        # Cache mechanism
        if use_cache:
            self.cache = FeatureCache(max_size=cache_size, cache_dim=d_inner)
        else:
            self.cache = None
        self.cache_weight = nn.Parameter(torch.tensor(0.5))
        self.similarity_threshold = 0.7

    def forward(self, x, cache_key=None):
        assert x is not None, "LoRACacheModule.forward input x is None"
        original_shape = x.shape
        if len(original_shape) == 4 and original_shape[1] == self.d_inner:
            x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        # 1. LoRA processing
        if self.use_lora:
            x_reshaped = x.view(B * H * W, C)
            x_lora_down = self.lora_down(x_reshaped)
            x_lora_up = self.lora_up(x_lora_down)
            x_lora = x_lora_up.view(B, H, W, C)
        else:
            x_lora = x
        if self.use_cache and self.cache is not None:
            cached_feats = self.cache.get_cached_features(x_lora, self.similarity_threshold)
            x_final_list = []
            for b in range(B):
                cached_feat, similarity = cached_feats[b]
                if cached_feat is not None:
                    if cached_feat.shape != x_lora[b:b+1].shape:
                        cached_feat = F.interpolate(
                            cached_feat.permute(0, 3, 1, 2),
                            size=(H, W),
                            mode='bilinear'
                        ).permute(0, 2, 3, 1)
                    cache_weight = torch.sigmoid(self.cache_weight)
                    x_fused = cache_weight * x_lora[b:b+1] + (1 - cache_weight) * cached_feat
                    x_reshaped = x_fused.view(H * W, C)
                    x_final = self.feature_fusion(
                        torch.cat([x_reshaped, x_lora_down[b*H*W:(b+1)*H*W]] if self.use_lora else [x_reshaped, x_reshaped], dim=-1)
                    ).view(1, H, W, C)
                else:
                    x_final = x_lora[b:b+1]
                assert x_final is not None, f"x_final is None at batch {b}"
                assert torch.is_tensor(x_final), f"x_final is not a tensor, but {type(x_final)}"
                x_final_list.append(x_final)
            assert len(x_final_list) > 0, "x_final_list is empty, which means batch_size=0 or the cache logic is有问题"
            assert all([x is not None for x in x_final_list]), "x_final_list has None, which means the cache logic is有问题"
            assert all([torch.is_tensor(x) for x in x_final_list]), "x_final_list has non-tensor elements"
            x_final = torch.cat(x_final_list, dim=0)
            self.cache.update(x_lora, cache_key)
        else:
            x_final = x_lora
        if len(original_shape) == 4 and original_shape[1] == self.d_inner:
            x_final = x_final.permute(0, 1, 3, 2)
        if not torch.is_tensor(x_final):
            raise RuntimeError(f"LoRACacheModule.forward return value is not a tensor, but {type(x_final)}")
        return x_final


class ViTScoreMap(nn.Module):
    
    def __init__(self, in_channels, patch_size=8, embed_dim=64, num_layers=1, num_heads=2):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Correctly define pos_embed as a learnable parameter in __init__
        self.pos_embed = nn.Parameter(torch.zeros(1, (224 // patch_size)**2, embed_dim)) # placeholder size
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] != self.patch_size and x.shape[1] != self.embed_dim:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, 'H, W must be divisible by patch_size'
        
        # use unfold to keep more details
        x_unfold = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = x_unfold.transpose(1, 2)  # [B, N_patches, C*patch_size*patch_size]
        
        # project to embed_dim
        if patches.shape[-1] != self.embed_dim:
            patches = nn.Linear(patches.shape[-1], self.embed_dim, device=x.device)(patches)
        
        N_patch = patches.shape[1]
        
        # adjust position encoding
        if self.pos_embed.shape[1] != N_patch:
            orig_N = self.pos_embed.shape[1]
            orig_size = int(orig_N ** 0.5)
            pos_embed = F.interpolate(
                self.pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2),
                size=(H // self.patch_size, W // self.patch_size),
                mode='bilinear'
            ).permute(0, 2, 3, 1).flatten(1, 2)
        else:
            pos_embed = self.pos_embed

        # reduce the impact of position encoding
        patches = patches + 0.1 * pos_embed

        feats = self.transformer(patches)
        scores = self.score_head(feats).squeeze(-1)
        
        # use sigmoid instead of softmax, keep more differences
        scores = torch.sigmoid(scores)
        
        # add content-related score calculation
        patch_variance = torch.var(patches, dim=-1)  # calculate the variance of the patch
        content_score = torch.sigmoid(patch_variance * 10)  # the patch with larger variance has higher score
        
        # combine the transformer score and the content score
        final_scores = 0.7 * scores + 0.3 * content_score
        
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        score_map = final_scores.view(B, H_patch, W_patch)
        return score_map


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # we use dilated-conv & DWConv for lightSR for a large ERF
            compress_ratio = 2
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 0),
                nn.Conv2d(num_feat//compress_ratio, num_feat // compress_ratio, 3, 1, 1,groups=num_feat//compress_ratio),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 0),
                nn.Conv2d(num_feat, num_feat, 3,1,padding=2,groups=num_feat,dilation=2),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else:
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class MLPAdapter(nn.Module):
    """
    MLP adapter, adapts MLP to a similar interface to CAB
    Handles [B, C, H, W] format input, internally converts to sequence format for MLP
    """
    def __init__(self, num_feat, hidden_features=None, drop=0.1):
        super(MLPAdapter, self).__init__()
        self.num_feat = num_feat
        hidden_features = hidden_features or num_feat * 2
        self.mlp = Mlp(num_feat, hidden_features, num_feat, drop=drop)
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Convert to sequence format
        x_seq = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_seq = x_seq.view(B, H * W, C)  # [B, H*W, C]
        # MLP processing
        x_seq = self.mlp(x_seq)  # [B, H*W, C]
        # Revert to convolutional format
        x_out = x_seq.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return x_out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SS2D(nn.Module):
    printed_LC = False  # static variable
    printed_ViT = False
    image_counter = 0  # global image counter
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            # LoRA+Cache params
            use_lora=True,
            lora_rank=16,
            use_cache=True,
            cache_size=5,
            # New parameters for adaptive scan
            use_adaptive_scan=True, # Set to True to enable your strategy
            score_map_patch_size=32,

            k=2,  # new parameter, control K
            direction='forward',  # new parameter, control the order when k=1
            
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        self.use_adaptive_scan = use_adaptive_scan
        self.use_lora = use_lora
        self.use_cache = use_cache
        self.score_map_patch_size = score_map_patch_size
        self.use_lora = use_lora
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()


        
        # Re-add LoRA and Cache module
        if not SS2D.printed_LC:
            if self.use_lora and self.use_cache:
                print("[LoRA+Cache] Both LoRA and Cache are enabled")
            elif self.use_lora and not self.use_cache:
                print("[LoRA+Cache] Only LoRA is enabled")
            elif not self.use_lora and self.use_cache:
                print("[LoRA+Cache] Only Cache is enabled")
            else:
                print("[LoRA+Cache] Both LoRA and Cache are disabled")
            SS2D.printed_LC = True

        if self.use_lora or self.use_cache:
            self.lora_cache = LoRACacheModule(
                d_inner=self.d_inner,
                lora_rank=lora_rank,
                cache_size=cache_size,
                use_cache=self.use_cache,
                use_lora=self.use_lora,
            )
        else:
            self.lora_cache = None



        
        if self.use_adaptive_scan:
            if not SS2D.printed_ViT:
                print("[VAMamba] ViTScoreMap is enabled")
                SS2D.printed_ViT = True
            self.vit_score_map = ViTScoreMap(
                in_channels=self.d_inner, 
                patch_size=self.score_map_patch_size,
                embed_dim=self.d_inner, 
                num_layers=1, 
                num_heads=4
            )
            self.K = k  # support K=1
            self.direction = direction  # new parameter
        else:
            if not SS2D.printed_ViT:
                print("[VAMamba] ViTScoreMap is disabled")
                SS2D.printed_ViT = True
            self.K = 4
            self.direction = 'forward'

        self.dt_rank = math.ceil(self.d_inner / 16) if dt_rank == "auto" else dt_rank
        
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for _ in range(self.K)
        ])
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = nn.ModuleList([
             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs) for _ in range(self.K)
        ])
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)

        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
    def adaptive_patch_traversal(self, score_map):
        H, W = score_map.shape
        device = score_map.device
        visited = torch.zeros_like(score_map, dtype=torch.bool, device=device)
        order = []
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while len(order) < H * W:
            # find the patch with the highest score as the starting point
            if len(order) == 0:
                # first time: find the global highest score
                start_idx = torch.argmax(score_map).item()
                start_i, start_j = divmod(start_idx, W)
            else:
                # later: find the patch with the highest score
                unvisited_scores = score_map[~visited]
                if unvisited_scores.numel() == 0:
                    break
                max_unvisited_score_idx = torch.argmax(unvisited_scores).item()
                unvisited_indices = (~visited).nonzero(as_tuple=False)
                start_i, start_j = unvisited_indices[max_unvisited_score_idx]
                # ensure to be Python integers
                start_i, start_j = start_i.item(), start_j.item()
            
            # greedy traversal from the starting point
            current_i, current_j = start_i, start_j
            visited[current_i, current_j] = True
            order.append((current_i, current_j))
            
            # greedy traversal: choose the patch with the highest score in the four directions
            while True:
                best_neighbor = None
                best_score = float('-inf')
                
                # check the four directions, handle the boundary cases correctly
                for di, dj in directions:
                    ni, nj = current_i + di, current_j + dj
                    # check the boundary and whether it has been visited
                    if 0 <= ni < H and 0 <= nj < W and not visited[ni, nj]:
                        if score_map[ni, nj] > best_score:
                            best_score = score_map[ni, nj]
                            best_neighbor = (ni, nj)
                
                # if the unvisited neighbor is found, continue the traversal
                if best_neighbor is not None:
                    current_i, current_j = best_neighbor
                    visited[current_i, current_j] = True
                    order.append((current_i, current_j))
                else:
                    # all four directions have been visited or reached the boundary, end the current path
                    break
        
        return order
        
    def forward_core_adaptive(self, x: torch.Tensor, scan_orders: list):
        B, C, H, W = x.shape
        p = self.score_map_patch_size
        H_patch, W_patch = H // p, W // p
        N_patches = H_patch * W_patch

        # 1. Get patch features by average pooling
        patch_features = F.adaptive_avg_pool2d(x, (H_patch, W_patch))
        patch_features = patch_features.view(B, C, -1) # [B, C, N_patches]
        
        # 2. Reorder sequences based on scan_orders
        reordered_features = torch.zeros_like(patch_features)
        for b in range(B):
            order_indices = torch.tensor([i * W_patch + j for i, j in scan_orders[b]], device=x.device, dtype=torch.long)
            reordered_features[b] = patch_features[b, :, order_indices]
            
        # 3. Create sequence(s)
        if self.K == 1:
            if self.direction == 'forward':
                xs = reordered_features.unsqueeze(1)  # [B, 1, C, N_patches]
            elif self.direction == 'backward':
                xs = torch.flip(reordered_features, dims=[-1]).unsqueeze(1)  # [B, 1, C, N_patches]
            else:
                raise ValueError(f"Unknown direction: {self.direction}, should be 'forward' or 'backward'")
        else:
            # When K > 1, create multiple sequences
            if self.K == 2:
                xs = torch.stack([reordered_features, torch.flip(reordered_features, dims=[-1])], dim=1)
            else:
                # For K > 2, create K sequences with different orderings
                sequences = [reordered_features]
                for k in range(1, self.K):
                    if k % 2 == 1:
                        sequences.append(torch.flip(reordered_features, dims=[-1]))
                    else:
                        sequences.append(reordered_features)
                xs = torch.stack(sequences, dim=1)

        # 4. Run SSM
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        
        out_y = self.selective_scan(
            xs.flatten(1, 2), dts.flatten(1, 2),
            self.A_logs.float(), Bs.float(), Cs.float(), self.Ds.float(), z=None,
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
        ).view(B, self.K, C, N_patches)

        # 5. Un-shuffle the output
        unshuffled_y = torch.zeros_like(out_y)
        for b in range(B):
            order_indices = torch.tensor([i * W_patch + j for i, j in scan_orders[b]], device=x.device, dtype=torch.long)
            inverse_order = torch.empty_like(order_indices)
            inverse_order[order_indices] = torch.arange(N_patches, device=x.device)
            unshuffled_y[b] = out_y[b, :, :, inverse_order]

        # 6. Fuse and reconstruct
        if self.K == 1:
            y_fused = unshuffled_y[:, 0]  # Only one sequence when K=1
        elif self.K == 2:
            y_fused = unshuffled_y[:, 0] + unshuffled_y[:, 1].flip(dims=[-1])
        else:
            # For K > 2, fuse all sequences
            y_fused = unshuffled_y[:, 0]  # Start with first sequence
            for k in range(1, self.K):
                if k % 2 == 1:
                    y_fused = y_fused + unshuffled_y[:, k].flip(dims=[-1])
                else:
                    y_fused = y_fused + unshuffled_y[:, k]
        y_map = y_fused.view(B, C, H_patch, W_patch)
        y_reconstructed = F.interpolate(y_map, size=(H, W), mode='nearest')
        
        return y_reconstructed

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = self.K

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (B, 4, C, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, cache_key=None, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        
        # Apply LoRA and Cache module if enabled (works for both scan types)
        if self.use_lora or self.use_cache:
            # The LoRACacheModule expects [B, H, W, C]
            x_for_lora = x.permute(0, 2, 3, 1).contiguous()
            x_processed = self.lora_cache(x_for_lora, cache_key)
            # Convert back to [B, C, H, W]
            x = x_processed.permute(0, 3, 1, 2).contiguous()

        if self.use_adaptive_scan:
            score_maps = self.vit_score_map(x.clone()) # Use clone to avoid in-place modification issues
            
            # only visualize the score map under certain conditions (to avoid repeated generation)
            if cache_key and "layer_0_block_0" in cache_key:  # generate once for each image in the first layer
                # use the global counter to track the image index
                image_idx = SS2D.image_counter
                SS2D.image_counter += 1
                
               
            
            scan_orders = [self.adaptive_patch_traversal(s_map) for s_map in score_maps]
            y = self.forward_core_adaptive(x, scan_orders)
            y = y.permute(0, 2, 3, 1).contiguous() # [B,H,W,C]
        else:
            y1, y2, y3, y4 = self.forward_core(x)
            assert y1.dtype == torch.float32
            y = y1 + y2 + y3 + y4
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

   


class RAMBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            use_lora = True,
            use_cache = True,
            lora_rank = 16,
            cache_size = 5,
            use_adaptive_scan = True,   
            score_map_patch_size=32,
            k = 2,
            direction = 'forward',
            **kwargs,
            

    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate,
                                   use_lora = use_lora,
                                   use_cache = use_cache,
                                   lora_rank = lora_rank,
                                   cache_size = cache_size,
                                   use_adaptive_scan = use_adaptive_scan,
                                   score_map_patch_size=score_map_patch_size,   
                                   k = k,
                                   direction = direction,
                                   **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim,is_light_sr)
        # self.conv_blk = MLPAdapter(hidden_dim, hidden_dim * 2, drop=0.1)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))



    def forward(self, input, x_size, cache_key=None):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x, cache_key=cache_key))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class BasicLayer(nn.Module):
    """ The Basic VAMamba Layer in one Residual Adaptive Mamba Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,is_light_sr=False,
                 use_lora = True,
                 lora_rank = 16,
                 use_cache = True,
                 cache_size = 5,
                 use_adaptive_scan = True,
                 score_map_patch_size=32,
                 k = 2,
                 direction = 'forward',
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(RAMBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution,is_light_sr=is_light_sr,
                use_lora = use_lora,
                use_cache = use_cache,
                lora_rank = lora_rank,
                cache_size = cache_size,
                use_adaptive_scan = use_adaptive_scan,
                score_map_patch_size=score_map_patch_size,
                k = k,
                direction = direction
                ))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, cache_key=None):
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size, cache_key)
            else:
                block_cache_key = f"{cache_key}_block_{i}" if cache_key is not None else f"block_{i}"
                x = blk(x, x_size, block_cache_key)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


@ARCH_REGISTRY.register()
class VAMamba(nn.Module):
    r"""VAMamba Model

    A PyTorch implementation of "A Visual Adaptive Mamba for Image Restoration".

    Args:
    - img_size (int | tuple[int], default=64): Input image size (H, W). Used by `PatchEmbed`/`PatchUnEmbed`.
    - patch_size (int | tuple[int], default=1): Patch size. A value of 1 means per-pixel embedding.
    - in_chans (int, default=3): Number of input image channels.
    - embed_dim (int, default=96): Backbone feature dimension / token embedding channels.
    - depths (tuple[int], default=(6, 6, 6, 6)): Number of RAMBlocks inside each Residual Group.
    - d_state (int, default=16): Hidden state size of the State Space Model (SSM).
    - mlp_ratio (float, default=2.0): Expansion ratio inside SS2D; `d_inner = mlp_ratio * d_model`.
    - drop_rate (float, default=0.0): Dropout probability applied after projections/positional operations.
    - drop_path_rate (float, default=0.1): Stochastic depth rate, linearly scheduled across blocks.
    - norm_layer (nn.Module, default=nn.LayerNorm): Normalization layer type.
    - patch_norm (bool, default=True): Whether to apply normalization after `PatchEmbed`.
    - use_checkpoint (bool, default=False): Enable gradient checkpointing to save memory.

    Reconstruction and residual options:
    - upscale (int, default=2): Upsampling factor for SR. Effective when `upsampler` is 'pixelshuffle' or 'pixelshuffledirect'.
    - img_range (float, default=1.0): Intensity range scaling for input/output.
    - upsampler (str, default=''): Reconstruction head type:
        * 'pixelshuffle': classical SR with stacked PixelShuffle.
        * 'pixelshuffledirect': lightweight one-step SR (1 conv + PixelShuffle).
        * otherwise/empty: denoising or same-resolution reconstruction with `conv_last`.
    - resi_connection (str, default='1conv'): Pre-residual conv block:
        * '1conv': single 3x3 convolution.
        * '3conv': lightweight three-layer conv sequence (3x3 → 1x1 → 3x3).

    LoRA / Cache / Adaptive scanning:
    - use_lora (bool, default=True): Enable LoRA low-rank adaptation on `d_inner` features in SS2D.
    - lora_rank (int, default=16): LoRA rank (low-rank dimension).
    - use_cache (bool, default=True): Enable feature caching and fusion across steps/batches.
    - cache_size (int, default=5): Maximum length of the cache queue.
    - use_adaptive_scan (bool, default=True): Enable content-adaptive scanning driven by `ViTScoreMap`.
    - score_map_patch_size (int, default=32): Patch size for score map; must divide feature-map H and W.
    - k (int, default=2): Number of scan sequences:
        * With adaptive scanning on: K=1/2/...; K=2 uses a forward + backward pair.
        * With adaptive scanning off: internally fixed to K=4 (four fixed paths).
    - direction (str, default='forward'): Scan direction when K=1; either 'forward' or 'backward'.

    Notes:
    - Fixed scan (use_adaptive_scan=False): Builds 4 fixed paths, merges them and runs a single SSM call in parallel, then untranspose/flip where needed and sum to fuse.
    - Adaptive scan (use_adaptive_scan=True): Builds content-aware orders from `ViTScoreMap`; when K>1, constructs forward/backward sequences, merges them and runs a single SSM call, then inverse-shuffles and fuses.
    - LoRA and Cache are available in both modes.
    """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state = 16,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 use_lora = True,
                 lora_rank =  16,
                 use_cache = True,
                 cache_size = 5,
                 use_adaptive_scan = True,
                 score_map_patch_size=32,
                 k = 2,
                 direction = 'forward',

                 **kwargs):
        super(VAMamba, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio=mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim


        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler=='pixelshuffledirect' else False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.use_lora = use_lora
        self.use_cache = use_cache
        self.lora_rank = lora_rank
        self.cache_size = cache_size
        self.use_adaptive_scan = use_adaptive_scan
        self.score_map_patch_size = score_map_patch_size
        self.k = k
        self.direction = direction
        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 6-layer
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state = d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr = self.is_light_sr,

                use_lora = self.use_lora,
                lora_rank = self.lora_rank,
                use_cache = self.use_cache,
                cache_size = self.cache_size,
                use_adaptive_scan = self.use_adaptive_scan,
                score_map_patch_size=self.score_map_patch_size,
                k = self.k,
                direction = self.direction
              
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, cache_key=None):
        x_size = (x.shape[2], x.shape[3])   
        x = self.patch_embed(x) # N,L,C
        x = self.pos_drop(x) 
        for i, layer in enumerate(self.layers):
            layer_cache_key = f"{cache_key}_layer_{i}" if cache_key is not None else f"layer_{i}"
            x = layer(x, x_size, layer_cache_key)
        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, cache_key=None):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':

            x = self.conv_first(x) 
            x = self.conv_after_body(self.forward_features(x, cache_key)) + x 
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':

            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, cache_key)) + x
            x = self.upsample(x)

        else:

            x_first = self.conv_first(x) 
            res = self.conv_after_body(self.forward_features(x_first, cache_key)) + x_first # 深层提取  核心
            x = x + self.conv_last(res)  

        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class ResidualGroup(nn.Module):
    """Residual Adaptive Mamba Group (RAMG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
        is_light_sr: Whether to use lightweight SR.
        use_lora: Whether to use LoRA.
        lora_rank: The rank of LoRA.
        use_cache: Whether to use cache.
        cache_size: The size of cache.
        use_adaptive_scan: Whether to use adaptive scanning.
        score_map_patch_size: The patch size of score map.
        k: The number of paths.
        direction: The direction of scanning.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 is_light_sr = False,
                 use_lora = True,
                 lora_rank = 16,
                 use_cache = True,
                 cache_size = 5,
                 use_adaptive_scan = True,
                 score_map_patch_size=32,
                 k = 2,
                 direction = 'forward',
                 ):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr,
            use_lora = use_lora,
            lora_rank = lora_rank,
            use_cache = use_cache,
            cache_size = cache_size,
            use_adaptive_scan = use_adaptive_scan,
            score_map_patch_size=score_map_patch_size,
            k = k,
            direction = direction
            )

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size, cache_key=None):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, cache_key), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops



class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

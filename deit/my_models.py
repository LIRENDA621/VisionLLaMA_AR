# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, Block, Attention, PatchEmbed, Mlp, DropPath
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, to_2tuple
from pos_embed import get_2d_sincos_pos_embed
import random
from utils import get_scale, get_2d_sincos_pos_embed
from typing import Optional, Tuple
from torch.nn import functional as F

try:
    from lightning_utilities.core.imports import RequirementCache
    FlashAttention2Available = bool(RequirementCache("flash-attn>=2.0.0.post1"))
except ImportError:
    FlashAttention2Available = False


class MyPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)

class ConvPatchEmbed(nn.Module):
    """ Image to Patch Embedding using more convolutions
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, enable_rpe=True):
        super(RBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale,  drop, attn_drop, drop_path, act_layer,
                                     norm_layer, enable_rpe)
        del self.attn
        self.attn = RAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, enable_rpe=enable_rpe)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

class LLaMAMLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        n_embd = in_features
        hidden_dim = 4 * n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        self.c_fc1 = nn.Linear(n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RBlockRMSNorm(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=RMSNorm, enable_rpe=True, rpe_ratio=1.0, token_num=None, 
                 casual_attn_1d=False, casual_attn_2d=False, use_cls_token=True):
        super(RBlockRMSNorm, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale,  drop, attn_drop, drop_path, act_layer,
                                     norm_layer)
        del self.attn
        self.attn = RAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                               enable_rpe=enable_rpe, rpe_ratio=rpe_ratio, token_num=token_num, casual_attn_1d=casual_attn_1d, casual_attn_2d=casual_attn_2d,
                               use_cls_token=use_cls_token)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LSRBlockRMSNorm(RBlockRMSNorm):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=RMSNorm, init_values=1e-4, enable_rpe=True, rpe_ratio=1.0,
                 token_num=None, casual_attn_1d=False, casual_attn_2d=False, use_cls_token=True):
        super(LSRBlockRMSNorm, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
                                              act_layer, norm_layer, enable_rpe, rpe_ratio, token_num, casual_attn_1d, casual_attn_2d,
                                              use_cls_token)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class LAMABlock(LSRBlockRMSNorm):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=RMSNorm, init_values=1e-4,  enable_rpe=True, rpe_ratio=1.0, token_num=None, 
                 casual_attn_1d=False, casual_attn_2d=False, use_cls_token=True):
        super(LAMABlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer,
                         norm_layer, init_values, enable_rpe, rpe_ratio, token_num, casual_attn_1d, casual_attn_2d, use_cls_token)
        del self.mlp
        self.mlp = LLaMAMLP(in_features=dim,  act_layer=act_layer, drop=drop)


class LAMABlockNoLS(RBlockRMSNorm):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=RMSNorm, enable_rpe=True):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer, enable_rpe)
        del self.mlp
        self.mlp = LLaMAMLP(in_features=dim, act_layer=act_layer, drop=drop)


class VisionTransformerMoreLE(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, blk_cls=Block, le_numb=6):
        super(VisionTransformerMoreLE, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                                                      num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate,
                                                      attn_drop_rate,
                                                      drop_path_rate, hybrid_backbone, norm_layer)
        self.le_num = le_numb

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        for i, blk in enumerate(self.blocks):
            if i < self.le_num:
                x = x + self.pos_embed
                x = self.pos_drop(x)
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, scale=1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # Hc/2
    t = torch.arange(end)  # type: ignore
    t = t * scale
    freqs = torch.outer(t, freqs).float()  # N,Hc/2 type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 N,Hc/2
    return freqs_cis #

def precompute_freqs_cis_2d(dim: int, end: int, theta: float = 10000.0, scale=1.0, use_cls=False, cls_pos=-1):
    H = int( end**0.5 )
    # assert  H * H == end
    flat_patch_pos = torch.arange(0 if not use_cls else -1, end) # N = end
    x_pos = flat_patch_pos % H # N
    y_pos = flat_patch_pos // H # N
    x_pos = x_pos * scale
    y_pos = y_pos * scale
    if use_cls and cls_pos !=-1:
        x_pos[0] = cls_pos
        y_pos[0] = cls_pos
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(end if not use_cls else end + 1, -1)
    # we need to think how to implement this for multi heads.
    # freqs_cis = torch.cat([x_cis, y_cis], dim=-1) # N, Hc/2
    return freqs_cis


def precompute_freqs_cis_2d_general(dim: int,  end: int = 1, step:int =1,  W=0, bias=0.0,  theta: float = 10000.0, scale=1.0, use_cls=False):
    if W == 0:
        W = int( end**0.5 )
    # assert  H * H == end
    flat_patch_pos = torch.arange(0 if not use_cls else -1, end) # N = end
    x_pos = flat_patch_pos % W # N
    x_pos = bias + x_pos * step
    y_pos = flat_patch_pos // W # N
    y_pos = bias + y_pos * step
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_pos = scale * x_pos
    y_pos = scale * y_pos
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(end if not use_cls else end + 1, -1)
    # we need to think how to implement this for multi heads.
    # freqs_cis = torch.cat([x_cis, y_cis], dim=-1) # N, Hc/2
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # x: B N H Hc/2
    # freqs_cis:  N, H*Hc/2 or  N Hc/2
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if freqs_cis.shape[-1] == x.shape[-1] and ndim == 4:
        shape = [1 if i == 2 or i == 0 else d for i, d in enumerate(x.shape)]  # 1, N, 1, Hc/2
    elif freqs_cis.shape[-1] != x.shape[-1]:
        shape = [d if i != 0 else 1 for i, d in enumerate(x.shape)] # 1, N, H, Hc/2
        # B, N, Hc/2
    elif freqs_cis.shape[-1] == x.shape[-1] and ndim == 3:
        shape = [1 if i == 0 else d for i, d in enumerate(x.shape)]  # 1, N, 1, Hc/2

    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq : B N H Hc
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # B, N, H, Hc
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_rotary_emb_mhsa(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq : B N H Hc
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2) # B, N, H, Hc
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_single(
        xq: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq : B N H Hc
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # B, N, H, Hc
    return xq_out.type_as(xq)

def create_causal_mask(size):
        """
        Create a causal mask to apply causal attention.

        Args:
            size (int): The size of the mask (sequence length).

        Returns:
            torch.Tensor: The causal mask.
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)  # Create an upper triangular matrix
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask

class RAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., enable_rpe=True, rpe_ratio=1.0, token_num=None,
                 casual_attn_1d=False, casual_attn_2d=False, use_cls_token=True):
        super(RAttention, self).__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.enable_rpe = enable_rpe
        print('rpe_ratio :', rpe_ratio)
        self.rpe_ratio = rpe_ratio
        self.casual_attn_1d = casual_attn_1d
        self.casual_attn_2d = casual_attn_2d
        self.token_num = token_num
        
        if self.casual_attn_2d:
            assert token_num != None
            H = token_num ** 0.5
            flat_patch_pos = torch.arange(0, token_num)
            x_pos = flat_patch_pos // H
            y_pos = flat_patch_pos % H

            total_visible_pos = []
            for x, y in zip(x_pos, y_pos):
                x, y = int(x), int(y)
                visible_pos = []
                for row in range(x + 1):
                    for col in range(y + 1):
                        visible_pos.append(int(row * H + col))
                total_visible_pos.append(visible_pos)

            self.casual_mask = torch.full((token_num, token_num), float('-inf'))

            # 遍历 total_visible_pos 列表
            for i, pos_list in enumerate(total_visible_pos):
                self.casual_mask[i, pos_list] = 0
            
            if use_cls_token:
                new_mask = torch.full((token_num + 1, token_num + 1), float(0.))
                new_mask[1: token_num + 1, 1:token_num + 1] = self.casual_mask
                # new_mask[token_num, :] = 0
                self.casual_mask = new_mask
        elif self.casual_attn_1d:
            casual_mask = create_causal_mask(token_num)
            if use_cls_token:
                new_mask = torch.full((token_num + 1, token_num + 1), float(0.))
                new_mask[1: token_num + 1, 1:token_num + 1] = casual_mask
                # new_mask[token_num, :] = 0
                self.casual_mask = new_mask
            # attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)  # Apply causal mask



    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        if self.enable_rpe:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B N H Hc
            rpe_dim = int(self.rpe_ratio * C // self.num_heads)
            if rpe_dim > 0:
                q_rot, k_rot = apply_rotary_emb(q[..., :rpe_dim], k[..., :rpe_dim], freqs_cis=freqs_cis)
                q = torch.cat((q_rot, q[..., rpe_dim:]), dim=-1)
                k = torch.cat((k_rot, k[..., rpe_dim:]), dim=-1)

            q = q.transpose(1, 2) # B, H, N, Hc
            k = k.transpose(1, 2) # B, H, N, Hc
            v = v.transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.casual_attn_2d or self.casual_attn_1d:
                attn = attn + self.casual_mask.unsqueeze(0).unsqueeze(0).to(x.device)  # Apply causal mask
            attn = attn.softmax(dim=-1)

           

            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        else:
            return super().forward(x)

class MHSAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1, sr_ratio=1.0,
                 enable_rpe=True, rpe_ratio=1.0, use_cls_token=False, auto_scale=False, use_ntk=False, casual_attn=False):
        """
        ws 1 for stand attention
        """
        super(MHSAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.ws = ws
        self.enable_rpe = enable_rpe
        self.freqs_cis = None
        self.rpe_ratio = rpe_ratio
        self.use_cls_token = use_cls_token
        self.auto_scale = auto_scale
        self.casual_attn=casual_attn

    

    def forward(self, x, H, W, base_size=14):
        B, N, C = x.shape
        
        # self.freqs_cis = precompute_freqs_cis_2d(int(self.dim // self.num_heads * self.rpe_ratio),
        #                                              self.ws*self.ws, use_cls=self.use_cls_token).to(x.device)
        if self.freqs_cis is None:
            self.freqs_cis = precompute_freqs_cis_2d(int(self.dim // self.num_heads * self.rpe_ratio),
                                                     N, use_cls=self.use_cls_token).to(x.device)
        else:
            if N != self.freqs_cis.shape[0]:
                self.freqs_cis = precompute_freqs_cis_2d(int(self.dim // self.num_heads * self.rpe_ratio),
                                                     N, use_cls=self.use_cls_token).to(x.device)
        
        

        x = x.view(B, H, W, C)

        # pad_l = pad_t = 0
        # pad_r = (self.ws - W % self.ws) % self.ws
        # pad_b = (self.ws - H % self.ws) % self.ws
        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # _, Hp, Wp, _ = x.shape
        # _h, _w = Hp // self.ws, Wp // self.ws
        # x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)  # B, _h, _w, ws, ws, C
        # qkv = self.qkv(x).reshape(B, _h * _w, self.ws * self.ws, 3, self.num_heads,
        #                                     C // self.num_heads).permute(3, 0, 2, 1, 4, 5) #B, _h*_w, ws*ws, 3, h, hc -> 3, B,  ws*ws, _h*_w, h,  hc,
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                            C // self.num_heads).permute(2, 0, 1, 3, 4) #B, _h*_w, ws*ws, 3, h, hc -> 3, B,  ws*ws, _h*_w, h,  hc,
        q, k, v = qkv[0], qkv[1], qkv[2] # B,  ws*ws, _h_w, h, hc.
        # q = q.reshape(B, self.ws * self.ws, -1, C // self.num_heads) # B,  ws*ws, _h_w * h, hc
        # k = k.reshape(B, self.ws * self.ws, -1, C // self.num_heads) # B,  ws*ws, _h_w * h, hc
        # v = v.reshape(B, self.ws * self.ws, -1, C // self.num_heads) # B,  ws*ws, _h_w * h, hc
        q = q.reshape(B * self.num_heads, -1, C // self.num_heads) # B,  ws*ws, _h_w * h, hc
        k = k.reshape(B * self.num_heads, -1, C // self.num_heads) # B,  ws*ws, _h_w * h, hc
        v = v.reshape(B * self.num_heads, -1, C // self.num_heads) # B,  ws*ws, _h_w * h, hc

        rpe_dim = int(self.rpe_ratio * C // self.num_heads)
        if rpe_dim > 0:
            q_rot, k_rot = apply_rotary_emb_mhsa(q[..., :rpe_dim], k[..., :rpe_dim], freqs_cis=self.freqs_cis)
            q = torch.cat((q_rot, q[..., rpe_dim:]), dim=-1) 
            k = torch.cat((k_rot, k[..., rpe_dim:]), dim=-1)
    
        # q = q.transpose(1, 2)  # B, _h_w * h, ws*ws, Hc 
        # k = k.transpose(1, 2)  # B, _h_w * h, ws*ws, Hc
        # v = v.transpose(1, 2) # B, _h_w * h, ws*ws, Hc

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.casual_attn:
            causal_mask = create_causal_mask(H * W).to(x.device)
            attn = attn + causal_mask.unsqueeze(0) # Apply causal mask

        attn = attn.softmax(dim=-1) # B, _h_w * h, ws*ws, ws*ws
        attn = self.attn_drop(attn)

        # attn = (attn @ v).reshape(B, -1, self.num_heads, self.ws*self.ws, C // self.num_heads) # B, _h_w,  h, ws*ws, Hc
        # attn = attn.transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C) # B, _h_w, ws*ws, C
        # x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        # if pad_r > 0 or pad_b > 0:
        #     x = x[:, :H, :W, :].contiguous()
        x = attn @ v # (b * h, N, C//h)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GroupAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1, sr_ratio=1.0,
                 enable_rpe=True, rpe_ratio=1.0, use_cls_token=False, auto_scale=False, use_ntk=False, casual_attn=False):
        """
        ws 1 for stand attention
        """
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws
        self.enable_rpe = enable_rpe
        self.freqs_cis = None
        self.rpe_ratio = rpe_ratio
        self.use_cls_token = use_cls_token
        self.auto_scale = auto_scale
        self.casual_attn=casual_attn

    

    def forward(self, x, H, W, base_size=14):
        B, N, C = x.shape
        if self.freqs_cis is None:
            self.freqs_cis = precompute_freqs_cis_2d(int(self.dim // self.num_heads * self.rpe_ratio),
                                                     self.ws*self.ws, use_cls=self.use_cls_token).to(x.device)
        else:
            if N != self.freqs_cis.shape[0]:
                self.freqs_cis = precompute_freqs_cis_2d(int(self.dim // self.num_heads * self.rpe_ratio),
                                                         self.ws*self.ws, use_cls=self.use_cls_token).to(x.device)

        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)  # B, Hp, Wp, C -> B, _h, _w, ws, ws, C
        qkv = self.qkv(x).reshape(B, _h * _w, self.ws * self.ws, 3, self.num_heads,
                                            C // self.num_heads).permute(3, 0, 2, 1, 4, 5) #B, _h*_w, ws*ws, 3, h, hc -> 3, B,  ws*ws, _h*_w, h,  hc,
        q, k, v = qkv[0], qkv[1], qkv[2] # B,  ws*ws, _h_w, h, hc.
        q = q.reshape(B, self.ws * self.ws, -1, C // self.num_heads) # B,  ws*ws, _h_w * h, hc
        k = k.reshape(B, self.ws * self.ws, -1, C // self.num_heads) # B,  ws*ws, _h_w * h, hc
        v = v.reshape(B, self.ws * self.ws, -1, C // self.num_heads) # B,  ws*ws, _h_w * h, hc

        rpe_dim = int(self.rpe_ratio * C // self.num_heads)
        if rpe_dim > 0:
            q_rot, k_rot = apply_rotary_emb(q[..., :rpe_dim], k[..., :rpe_dim], freqs_cis=self.freqs_cis)
            q = torch.cat((q_rot, q[..., rpe_dim:]), dim=-1) 
            k = torch.cat((k_rot, k[..., rpe_dim:]), dim=-1)
    
        q = q.transpose(1, 2)  # B, _h_w * h, ws*ws, Hc 
        k = k.transpose(1, 2)  # B, _h_w * h, ws*ws, Hc
        v = v.transpose(1, 2) # B, _h_w * h, ws*ws, Hc

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.casual_attn:
            causal_mask = create_causal_mask(self.ws * self.ws).to(x.device)
            attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)  # Apply causal mask

        attn = attn.softmax(dim=-1) # B, _h_w * h, ws*ws, ws*ws
        attn = self.attn_drop(attn)

        attn = (attn @ v).reshape(B, -1, self.num_heads, self.ws*self.ws, C // self.num_heads) # B, _h_w,  h, ws*ws, Hc
        attn = attn.transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C) # B, _h_w, ws*ws, C
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GSAttention(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,  sr_ratio=1,
                 enable_rpe=True, rpe_ratio=1.0, use_cls_token=False, auto_scale=False, use_ntk=False, casual_attn=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.enable_rpe = enable_rpe
        self.rpe_ratio = rpe_ratio
        self.use_cls_token = use_cls_token
        self.q_freqs_cis = None
        self.k_freqs_cis = None
        self.auto_scale = auto_scale # we should scale the t in RoPE when this is enabled
        self.use_ntk = use_ntk
        self.casual_attn=casual_attn

    def forward(self, x, H, W, base_size=14):
        B, N, C = x.shape
        N_reduce = int(H * W // self.sr_ratio // self.sr_ratio)
        rpe_dim = int(self.dim // self.num_heads * self.rpe_ratio)
        if self.auto_scale:
            scale = base_size / H
        else:
            scale = 1.0
        theta = 10000.0
        if self.use_ntk is not False:
            scale = 1.0
            theta = theta * self.use_ntk

        if self.q_freqs_cis is None and rpe_dim > 0:
            self.q_freqs_cis = precompute_freqs_cis_2d_general(rpe_dim, N, W=W, theta=theta, use_cls=self.use_cls_token, scale=scale).to(x.device)
        else:
            if N != self.q_freqs_cis.shape[0]:
                self.q_freqs_cis = precompute_freqs_cis_2d_general(rpe_dim, N, W=W, theta=theta, use_cls=self.use_cls_token, scale=scale).to(x.device)

        if self.k_freqs_cis is None and rpe_dim > 0:
            self.k_freqs_cis = precompute_freqs_cis_2d_general(rpe_dim, N_reduce, W=W,  step=self.sr_ratio, bias=1.0 - 1.0 / self.sr_ratio,
                                                               use_cls=self.use_cls_token, scale=scale).to(x.device)
        else:
            if N // self.sr_ratio**2 != self.k_freqs_cis.shape[0]:
                self.k_freqs_cis = precompute_freqs_cis_2d_general(rpe_dim, N_reduce, W=W, step=self.sr_ratio, bias=1.0 - 1.0 / self.sr_ratio,
                                                           use_cls=self.use_cls_token, scale=scale).to(x.device)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads) # B, N, H, Hc

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) # B, M, C
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4) # B,M,2C -> B,M, 2, H, Hc -> 2, B, M,  H, Hc
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        k, v = kv[0], kv[1] # B,  M, H, Hc

        if rpe_dim > 0:
            q_rot = apply_rotary_emb_single(q[..., :rpe_dim], freqs_cis=self.q_freqs_cis)
            k_rot = apply_rotary_emb_single(k[..., :rpe_dim], freqs_cis=self.k_freqs_cis)
            q = torch.cat((q_rot, q[..., rpe_dim:]), dim=-1)
            k = torch.cat((k_rot, k[..., rpe_dim:]), dim=-1)

        q = q.transpose(1, 2)  # B, H, N, Hc
        k = k.transpose(1, 2)  # B, H, M, Hc
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale


        if self.casual_attn:
            # flat_patch_pos = torch.arange(0, H * W)
            # x_pos = flat_patch_pos // (self.sr_ratio * H)

            # origin_x_pos = flat_patch_pos // H
            # origin_y_pos = flat_patch_pos % H

            # flat_patch_pos_for_y = torch.zeros(H * W)
            # for i, pos in enumerate(zip(origin_x_pos, origin_y_pos)):
            #     flat_patch_pos_for_y[i] = pos[0] + pos[1] * H
            # y_pos = flat_patch_pos_for_y // (self.sr_ratio * H)

            # token_visible_num = x_pos * self.sr_ratio + y_pos + 1 
            # causal_mask = torch.full((H * W, N_reduce), float('-inf')).to(x.device)
            # for i in range(H * W):
            #     causal_mask[i, :int(token_visible_num[i])] = 0.

            flat_patch_pos = torch.arange(0, H * W, device=x.device)
            x_pos = flat_patch_pos // (self.sr_ratio * H)

            origin_x_pos = flat_patch_pos // H
            origin_y_pos = flat_patch_pos % H

            flat_patch_pos_for_y = origin_x_pos + origin_y_pos * H
            y_pos = flat_patch_pos_for_y // (self.sr_ratio * H)

            token_visible_num = x_pos * self.sr_ratio + y_pos + 1
            token_visible_num = token_visible_num.long()  # Ensure it's long type for indexing

            # Initialize the causal mask with -inf and use broadcasting to fill in the visible tokens
            causal_mask = torch.full((H * W, N_reduce), float('-inf'), device=x.device)
            row_indices = torch.arange(H * W, device=x.device).unsqueeze(1)
            col_indices = torch.arange(N_reduce, device=x.device).unsqueeze(0)
            mask = col_indices < token_visible_num.unsqueeze(1)
            causal_mask[mask] = 0.

            # causal_mask = create_causal_mask(self.ws * self.ws).to(x.device)
            attn = attn + causal_mask.unsqueeze(0)  # Apply causal mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class GroupBlock(LAMABlock):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=RMSNorm, init_values=1e-4,  enable_rpe=True, rpe_ratio=1.0,
                 use_cls_token=False, sr_ratio=1, ws=1, auto_scale=False, use_ntk=False, casual_attn=False):
        super(GroupBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                         drop_path, act_layer, norm_layer, init_values, enable_rpe, rpe_ratio)
        del self.attn
        # if ws == 1 and not casual_attn:
        #     self.attn = GSAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio,
        #                             enable_rpe=enable_rpe, rpe_ratio=rpe_ratio, use_cls_token=use_cls_token,
        #                             auto_scale=auto_scale, use_ntk=use_ntk, casual_attn=casual_attn)
        # elif not casual_attn:
        #     self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws, enable_rpe=enable_rpe,
        #                                rpe_ratio=rpe_ratio, use_cls_token=use_cls_token, auto_scale=auto_scale, use_ntk=use_ntk, 
        #                                casual_attn=casual_attn)
        # elif casual_attn:
        #     self.attn = MHSAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio,
        #                             enable_rpe=enable_rpe, rpe_ratio=rpe_ratio, use_cls_token=use_cls_token,
        #                             auto_scale=auto_scale, use_ntk=use_ntk, casual_attn=casual_attn)
        if ws == 1:
            self.attn = GSAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio,
                                    enable_rpe=enable_rpe, rpe_ratio=rpe_ratio, use_cls_token=use_cls_token,
                                    auto_scale=auto_scale, use_ntk=use_ntk, casual_attn=casual_attn)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws, enable_rpe=enable_rpe,
                                       rpe_ratio=rpe_ratio, use_cls_token=use_cls_token, auto_scale=auto_scale, use_ntk=use_ntk, 
                                       casual_attn=casual_attn)


    def forward(self, x, H, W):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class GroupBlockWoLS(GroupBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=RMSNorm, init_values=1e-4,  enable_rpe=True, rpe_ratio=1.0,
                 use_cls_token=False, sr_ratio=1, ws=1, auto_scale=False):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer,
                         norm_layer, init_values, enable_rpe, rpe_ratio, use_cls_token, sr_ratio, ws, auto_scale)
        del self.gamma_1
        del self.gamma_2


    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GQA(nn.Module):
    """
    Grouped Query Attetnion https://arxiv.org/pdf/2305.13245.pdf
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., enable_rpe=True, rpe_ratio=1.0, query_groups=1):

        super().__init__()
        self.num_heads = num_heads
        self.query_groups = query_groups
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.rpe_ratio = rpe_ratio
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        shape = (num_heads + 2 * query_groups) * head_dim
        self.qkv = nn.Linear(dim, shape, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.qkv(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.num_heads // self.query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.query_groups, total_qkv, self.head_dim)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # repeat k and v if necessary
        if self.query_groups != 1:
            # for MHA this is a no-op
            k = k.expand(B, self.query_groups, q_per_kv, T, self.head_dim)
            v = v.expand(B, self.query_groups, q_per_kv, T, self.head_dim)

        q = q.reshape(B, -1, T, self.head_dim)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.head_dim)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.head_dim)  # (B, nh_v, T, hs)

        rope_dim = int(self.head_dim * self.rpe_ratio)
        q_roped,  k_roped= apply_rotary_emb(q[..., : rope_dim], k[..., : rope_dim], freqs_cis)
        q = torch.cat((q_roped, q[..., rope_dim :]), dim=-1)
        k = torch.cat((k_roped, k[..., rope_dim :]), dim=-1)

        y = self.scaled_dot_product_attention(q, k, v)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = self.head_dim ** -0.5

        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            # flash-attn requires (B, T, nh, hs)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=False)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)


class RAttention2D(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., enable_rpe=True):
        super(RAttention2D, self).__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.enable_rpe = enable_rpe

    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        if self.enable_rpe:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B N H Hc
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
            q = q.transpose(1, 2) # B, H, N, Hc
            k = k.transpose(1, 2) # B, H, N, Hc
            v = v.transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        else:
            return super().forward(x)


class VisionTransformerRotate(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, blk_cls=RBlock, scale=1.0,
                 rpe_flags=list(), stochastic_depth_decay=False, skip_first_dp=False, use_cls_token=True, casual_attn_1d=False, casual_attn_2d=False):
        super(VisionTransformerRotate, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                                                     num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate,
                                                     attn_drop_rate,
                                                     drop_path_rate, hybrid_backbone, norm_layer)
        if len(rpe_flags) == 0:
            rpe_flags = [True for i in range(depth)]
        print('rpe flags: ', rpe_flags)
        del self.blocks
        if stochastic_depth_decay:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        else:
            dpr = [drop_path_rate for i in range(depth)]
        if skip_first_dp:
            dpr[0] = 0.0
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.use_cls_token = use_cls_token
        self.token_num = img_size * img_size // (patch_size * patch_size)
        # self.token_num = img_size * img_size // (patch_size * patch_size) if not use_cls_token else img_size * img_size // (patch_size * patch_size) + 1 
        self.blocks = nn.ModuleList([blk_cls(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias,  qk_scale=qk_scale, drop=drop_rate,
                                             attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                                             enable_rpe=rpe_flags[i], token_num=self.token_num, 
                                             casual_attn_1d=casual_attn_1d, casual_attn_2d=casual_attn_2d, use_cls_token=self.use_cls_token)
                                     for i in range(depth)])
        self.apply(self._init_weights)
        self.freqs_cis = None
        self.num_heads = num_heads
        self.scale = scale

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.freqs_cis is None:
            self.freqs_cis = precompute_freqs_cis(
                self.embed_dim // self.num_heads, self.patch_embed.num_patches * 2, scale=self.scale
            ).to(x.device)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed
        x = self.pos_drop(x)
        freqs_cis = self.freqs_cis[0: self.patch_embed.num_patches, :]

        for blk in self.blocks:
            x = blk(x, freqs_cis)

        x = self.norm(x) # B N, C
        return x.mean(dim=1)

    def reset_scale(self, scale):
        print('reset scale ', scale)
        self.scale = scale
        if self.freqs_cis is not None:
            self.freqs_cis = precompute_freqs_cis(
                self.embed_dim // self.num_heads, self.patch_embed.num_patches * 2, scale=self.scale
            ).to(self.freqs_cis.device)

    def reset_ntk_scale(self, scale, factor=1.0):
        print('reset scale ', scale)
        self.scale = scale
        embed_dim = self.embed_dim // self.num_heads
        theta = 10000.0
        theta = theta * (
                (factor * scale) - (factor - 1)
        ) ** (embed_dim / (embed_dim - 2))

        if self.freqs_cis is not None:
            self.freqs_cis = precompute_freqs_cis(
                self.embed_dim // self.num_heads, self.patch_embed.num_patches * 2, scale=1.0, theta=theta
            ).to(self.freqs_cis.device)


class VisionLLaMA(VisionTransformerRotate):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, blk_cls=RBlock, scale=1.0,
                 rpe_flags=list(), stochastic_depth_decay=False, share_rpe=True, rpe_ratio=1.0, use_abs=False,
                 skip_first_dp=False, lr_pe=True, use_cls_token=False, theta=10000.0, auto_scale=False, cls_pos=-1,
                 casual_attn_1d=False, casual_attn_2d=False):
        """

        :param img_size:
        :param patch_size:
        :param in_chans:
        :param num_classes:
        :param embed_dim:
        :param depth:
        :param num_heads:
        :param mlp_ratio:
        :param qkv_bias:
        :param qk_scale:
        :param drop_rate:
        :param attn_drop_rate:
        :param drop_path_rate:
        :param hybrid_backbone:
        :param norm_layer:
        :param blk_cls:
        :param scale:
        :param rpe_flags:
        :param stochastic_depth_decay:
        :param share_rpe:
        :param rpe_ratio:
        :param use_abs: use abs positional encoding.
        :param skip_first_dp:
        :param lr_pe: use learnable PE if true else sin cos
        :param use_cls_token: use cls token or not
        """

        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
                         qk_scale, drop_rate, attn_drop_rate, drop_path_rate, hybrid_backbone, norm_layer,blk_cls, scale,
                         rpe_flags, stochastic_depth_decay, skip_first_dp, use_cls_token, casual_attn_1d, casual_attn_2d)
        self.share_rpe = share_rpe
        self.rpe_ratio = rpe_ratio
        self.use_abs = use_abs
        self.lr_pe = lr_pe
        self.use_cls_token = use_cls_token
        self.theta = theta
        self.auto_scale = auto_scale
        self.cls_pos = cls_pos
        

        if not use_cls_token:
            del self.cls_token
        if not self.use_abs:
            del self.pos_embed
        else:
            if not lr_pe: # we use fixed sin cos
                del self.pos_embed
                self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0)) # use 2d sin-cos PE.

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        if self.auto_scale:
            scale = 224 / H
        else:
            scale = 1.0

        if self.freqs_cis is None:
            if self.share_rpe:
                self.freqs_cis = precompute_freqs_cis_2d(int(self.embed_dim // self.num_heads * self.rpe_ratio),
                                                         self.patch_embed.num_patches, use_cls=self.use_cls_token,
                                                         theta=self.theta, scale=scale, cls_pos=self.cls_pos).to(x.device)
            else:
                self.freqs_cis = precompute_freqs_cis_2d(int(self.embed_dim * self.rpe_ratio),
                                                         self.patch_embed.num_patches, use_cls=self.use_cls_token,
                                                         theta=self.theta, scale=scale, cls_pos=self.cls_pos).to(x.device)
            # self.freqs_cis = self.freqs_cis.unsqueeze(0).repeat(x.shape[0], 1, 1)

        # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # # x = torch.cat((cls_tokens, x), dim=1)
        # # x = x + self.pos_embed
        if self.use_abs:
            if not self.use_cls_token:
                x = x + self.pos_embed[:, 1:]
            else:
                cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.pos_embed
        else:
            if self.use_cls_token:
                cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)
        num_patches = self.patch_embed.num_patches if not self.use_cls_token else self.patch_embed.num_patches + 1
        freqs_cis = self.freqs_cis[0: num_patches, :]

        for blk in self.blocks:
            x = blk(x, freqs_cis)

        if self.use_cls_token:
            x = self.norm(x) # B N, C
            return x[:, 0]
        else:
            return self.norm(x.mean(dim=1))
        # return x[:, 0] if self.use_cls_token else x.mean(dim=1)

    def reset_scale(self, scale):
        print('reset scale ', scale)
        self.scale = scale
        if self.freqs_cis is None:
            if self.share_rpe:
                self.freqs_cis = precompute_freqs_cis_2d(int(self.embed_dim // self.num_heads * self.rpe_ratio),
                                                         self.patch_embed.num_patches, use_cls=self.use_cls_token,
                                                         theta=self.theta)

            else:
                self.freqs_cis = precompute_freqs_cis_2d(int(self.embed_dim * self.rpe_ratio),
                                                         self.patch_embed.num_patches, use_cls=self.use_cls_token,
                                                         theta=self.theta)
            # self.freqs_cis = self.freqs_cis.unsqueeze(0).repeat(x.shape[0], 1, 1)


class PyramidVisionLLaMA(VisionLLaMA):
    """
    borrow the code from Twins (https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py)
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=[3, 6, 12, 24], mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, blk_cls=GroupBlock, scale=1.0,
                 rpe_flags=list(), stochastic_depth_decay=False, share_rpe=True, rpe_ratio=1.0, use_abs=False,
                 skip_first_dp=False, lr_pe=True, use_cls_token=False, embed_dims=[96, 192, 384, 768],
                 depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], auto_scale=False, use_peg=False, use_ntk=False,
                 casual_attn=False):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth, 1, mlp_ratio,
                         qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, hybrid_backbone, norm_layer,
                         blk_cls, scale, rpe_flags, stochastic_depth_decay, share_rpe, rpe_ratio, use_abs, skip_first_dp,
                         lr_pe, use_cls_token)
        del self.blocks

        if stochastic_depth_decay:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        else:
            dpr = [drop_path_rate for i in range(sum(depths))]

        # print('drop path list: ', dpr)
        print('use ntk: ', use_ntk)
        # self.wss = wss if not casual_attn else [1, 1, 1, 1]
        self.wss = wss


        self.depths = depths
        # transformer encoder
        cur = 0
        self.blocks = nn.ModuleList()
        self.patch_embeds = nn.ModuleList()

        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(MyPatchEmbed(img_size, patch_size, in_chans, embed_dims[i]))
            else:
                self.patch_embeds.append(
                    MyPatchEmbed(img_size // patch_size // 2 ** (i - 1), 2, embed_dims[i - 1], embed_dims[i]))

        for k in range(len(depths)):
            _block = nn.ModuleList([blk_cls(
                dim=embed_dims[k], num_heads=num_heads[k], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k], ws=1 if i % 2 == 1 else wss[k], use_cls_token=use_cls_token, # when casual_attn, wss[k]=1
                rpe_ratio=rpe_ratio, enable_rpe=True, auto_scale=auto_scale, use_ntk=use_ntk, casual_attn=casual_attn) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        self.use_peg = use_peg
        if self.use_peg:
            self.pegs = nn.ModuleList()
            for _embed_dim in embed_dims:
                self.pegs.append(PEGv1(_embed_dim, _embed_dim))
        self.apply(self._init_weights)
        del self.patch_embed
        # del self.cls_token

    def _init_weights(self, m):
        import math
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0 and self.use_peg:
                    x = self.pegs[i](x, H, W)

            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.norm(x) # B N, C

        if self.use_cls_token:
            x = self.norm(x) # B N, C
            return x[:, 0]
        else:
            return self.norm(x.mean(dim=1))


class VisionTransformerDePos(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, blk_cls=Block):
        super(VisionTransformerDePos, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                                                     num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate,
                                                     attn_drop_rate,
                                                     drop_path_rate, hybrid_backbone, norm_layer)
        del self.blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([blk_cls(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                             attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                                     for i in range(depth)])

        del self.pos_embed

        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]


class VisionTransformerGAP(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super(VisionTransformerGAP, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                                                   num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                                                   drop_path_rate, hybrid_backbone, norm_layer)

        del self.cls_token
        del self.pos_embed
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x.mean(dim=1)


@register_model
def deit_small_gap_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformerGAP(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vit_llama_rope_large_patch16(pretrained=False, **kwargs):
    model = VisionLLaMA(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), blk_cls=LAMABlock, share_rpe=True,
        use_cls_token=True, use_abs=False,  **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_llama_rope_large_as_patch16(pretrained=False, **kwargs):
    model = VisionLLaMA(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), blk_cls=LAMABlock, share_rpe=True,
        use_cls_token=True, use_abs=False, auto_scale=True,  **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_llama_rope_small_patch16(pretrained=False, **kwargs):
    model = VisionLLaMA(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), blk_cls=LAMABlock, share_rpe=True,
        use_cls_token=True, use_abs=False, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_llama_rope_base_patch16(pretrained=False, **kwargs):
    model = VisionLLaMA(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), blk_cls=LAMABlock, share_rpe=True,
        use_cls_token=True, use_abs=False,  **kwargs)
    model.default_cfg = _cfg()
    return model



@register_model
def pllama_wols_small_patch16(pretrained=False, **kwargs):
    model = PyramidVisionLLaMA(
        patch_size=4, embed_dim=512, depth=12, num_heads=[2, 4, 8, 16], mlp_ratio=4, qkv_bias=True,
        blk_cls=GroupBlock, share_rpe=True, use_cls_token=False, use_abs=False, stochastic_depth_decay=True,
        embed_dims=[64, 128, 256, 512], depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


# @register_model
# def pllama_wols_small_nkt10_patch16(pretrained=False, **kwargs):
#     model = PyramidVisionLLaMA(
#         patch_size=4, embed_dim=512, depth=12, num_heads=[2, 4, 8, 16], mlp_ratio=4, qkv_bias=True,
#         blk_cls=GroupBlock, share_rpe=True, use_cls_token=False, use_abs=False, stochastic_depth_decay=True,
#         embed_dims=[64, 128, 256, 512], depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], use_ntk=10,
#         **kwargs)
#     model.default_cfg = _cfg()
#     return model

# @register_model
# def pllama_wols_small_nkt5_patch16(pretrained=False, **kwargs):
#     model = PyramidVisionLLaMA(
#         patch_size=4, embed_dim=512, depth=12, num_heads=[2, 4, 8, 16], mlp_ratio=4, qkv_bias=True,
#         blk_cls=GroupBlock, share_rpe=True, use_cls_token=False, use_abs=False, stochastic_depth_decay=True,
#         embed_dims=[64, 128, 256, 512], depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], use_ntk=5,
#         **kwargs)
#     model.default_cfg = _cfg()
#     return model
#
# @register_model
# def pllama_wols_small_nkt2_patch16(pretrained=False, **kwargs):
#     model = PyramidVisionLLaMA(
#         patch_size=4, embed_dim=512, depth=12, num_heads=[2, 4, 8, 16], mlp_ratio=4, qkv_bias=True,
#         blk_cls=GroupBlock, share_rpe=True, use_cls_token=False, use_abs=False, stochastic_depth_decay=True,
#         embed_dims=[64, 128, 256, 512], depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], use_ntk=2,
#         **kwargs)
#     model.default_cfg = _cfg()
#     return model
#


# @register_model
# def pllama_wols_peg_small_patch16(pretrained=False, **kwargs):
#     model = PyramidVisionLLaMA(
#         patch_size=4, embed_dim=512, depth=12, num_heads=[2, 4, 8, 16], mlp_ratio=4, qkv_bias=True,
#         blk_cls=GroupBlock, share_rpe=True, use_cls_token=False, use_abs=False, stochastic_depth_decay=True,
#         embed_dims=[64, 128, 256, 512], depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],  use_peg=True,
#         **kwargs)
#     model.default_cfg = _cfg()
#     return model

@register_model
def pllama_wols_base_patch16(pretrained=False, **kwargs):
    model = PyramidVisionLLaMA(
        patch_size=4, embed_dim=768, depth=12, num_heads=[3, 6, 12, 24], mlp_ratio=4, qkv_bias=True,
        blk_cls=GroupBlock, share_rpe=True, use_cls_token=False, use_abs=False, stochastic_depth_decay=True,
        embed_dims=[96, 192, 384, 768], depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def pllama_wols_large_patch16(pretrained=False, **kwargs):
    model = PyramidVisionLLaMA(
        patch_size=4, embed_dim=1024, depth=12, num_heads=[4, 8, 16, 32], mlp_ratio=4, qkv_bias=True,
        blk_cls=GroupBlock, share_rpe=True, use_cls_token=False, use_abs=False, stochastic_depth_decay=True,
        embed_dims=[128, 256, 512, 1024], depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)

    model.default_cfg = _cfg()
    return model

@register_model
def pllama_wols_large_as_patch16(pretrained=False, **kwargs):
    model = PyramidVisionLLaMA(
        patch_size=4, embed_dim=1024, depth=12, num_heads=[4, 8, 16, 32], mlp_ratio=4, qkv_bias=True,
        blk_cls=GroupBlock, share_rpe=True, use_cls_token=False, use_abs=False, stochastic_depth_decay=True,
        embed_dims=[128, 256, 512, 1024], depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], auto_scale=True,
        **kwargs)

    model.default_cfg = _cfg()
    return model


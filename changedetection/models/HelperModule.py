import sys
import os
sys.path.append("/data1/wuxiaomeng/code/HGLRMamba/")
import torch
from classification.models.vmamba import  VSSBlock, Permute
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import DropPath
from classification.models.vmamba import Mlp

class LocalContextRefine(nn.Module):

    def __init__(self, in_features, reduce_ratio=1,r:int=2, act_layer=nn.GELU) -> None:
        super().__init__()
        d_inner = int(in_features // r)
        # d_inner = in_features
        self.norm = nn.BatchNorm2d(in_features)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=7, padding=3, groups=in_features),
            act_layer(),
            nn.BatchNorm2d(in_features),
        )
        self.local_ctx1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=d_inner, kernel_size=1, stride=1),
            act_layer(),
            nn.Conv2d(in_channels=d_inner, out_channels=in_features, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_features),
        )
        self.local_ctx2 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=d_inner, kernel_size=3, padding=1, stride=1),
            act_layer(),
            nn.Conv2d(in_channels=d_inner, out_channels=in_features, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_features),
        )
        self.channel_aware_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_features, out_channels=d_inner, kernel_size=1, padding=0, stride=1),
            act_layer(),
            nn.Conv2d(in_channels=d_inner, out_channels=in_features, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(in_features),
        )
        self.act = act_layer()
        self.gamma2 = ElementScale(in_features, init_value=1e-4)
        self.reduce_conv = nn.Sequential(
            nn.BatchNorm2d(in_features),
        )
        if reduce_ratio > 1:
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(in_features, in_features // reduce_ratio, kernel_size=1, stride=1, padding=0),
                act_layer(),
                nn.BatchNorm2d(in_features // reduce_ratio),
            )
    def forward(self, x_in):
        # x is B C H W
        B, C, H, W = x_in.shape
        x = self.in_conv(x_in)
        # short_cut = x
        x1 = self.local_ctx1(x)
        x2 = self.local_ctx2(x)
        x_s = self.norm(x1 + x2)
        x_channel_avg = self.channel_aware_avg(x_s)
        x_s = x + x1 + x2 + x_channel_avg

        pooled, indices = F.max_pool2d(x_s, kernel_size=2, stride=2, padding=0, return_indices=True)
        unpool = F.max_unpool2d(pooled, kernel_size=2, stride=2, padding=0,indices=indices)
        out = self.gamma2(torch.sigmoid(x_s + unpool)) * x_s + x_in


        return self.reduce_conv(out)


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class DSRAttention(nn.Module):

    def __init__(self, dim=128, num_heads=4,  qkv_bias=True, act_layer=nn.GELU) -> None:
        super(DSRAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.act_layer = act_layer()
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x:torch.Tensor, change_map:torch.Tensor):
        b, l, d = x.shape
        _, l2, _ = change_map.shape
        h = w = int(l ** 0.5)
        q = self.act_layer(self.to_q(x))
        k, v = self.to_kv(change_map).chunk(2, dim=-1)
        k, v = tuple(map(lambda x: self.act_layer(x), [k, v]))
        # v shape is b l d
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), [q, k, v])
        weight = F.softmax(q @ k.transpose(-1, -2) * self.dim ** -0.5, dim=-1)
        out = weight @ v
        out = rearrange(out, 'b h l d -> b l (h d)')


        return out

class DeepSemanticRefineBlock(nn.Module):

    def __init__(self, dim=128, num_heads=8, qkv_bias=True, changeguide_ratio=1., d_conv=3,
                 act_layer=nn.SiLU, mlp_ratio=4., drop=0., drop_path=0.) -> None:
        super().__init__()
        self.d_inner = int(dim * changeguide_ratio)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, self.d_inner)
        self.change_in_proj = nn.Linear(dim, self.d_inner)
        self.with_conv = d_conv > 0
        if self.with_conv:
            self.dwconv = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                groups=dim,
                bias=False,
            )
        self.attn = DSRAttention(dim=self.d_inner, num_heads=num_heads, qkv_bias=qkv_bias, act_layer=act_layer)
        self.act = act_layer()
        self.out_proj = nn.Linear(self.d_inner, dim)
        self.scale = ElementScale(embed_dims=dim, init_value=1e-4)

        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = nn.LayerNorm(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim, act_layer=act_layer, drop=drop)


    def forward(self, x_in:torch.Tensor, change_map:torch.Tensor):
        b, c, h, w = x_in.shape
        x = self.norm(x_in.view(b, c, -1).permute(0, 2, 1))
        x = self.in_proj(x)
        change_map = self.change_in_proj(change_map.view(b, c, -1).permute(0, 2, 1))
        act_res = self.act(x)
        if self.with_conv:
            x = self.dwconv(x.permute(0, 2, 1).view(b, self.d_inner, h, w))
            att = self.attn(x.view(b, self.d_inner, -1).permute(0, 2, 1), change_map)
        else:
            att = self.attn(x, change_map)
        z = self.drop_path(self.out_proj(att + act_res))

        if self.mlp_branch:
            z = self.norm2(z)
            z = self.drop_path(self.mlp(z))
        z = z.permute(0, 2, 1).view(b, c, h, w)
        return self.scale(z) + x_in


class GatedFusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, drop_path=0., layer_scale_init_value=1e-4, expand_ratio=3.):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
        )


        self.dwconv = nn.Conv2d(out_dim, out_dim, kernel_size=7, padding=3, groups=out_dim) # depthwise conv
        self.f1 = nn.Conv2d(out_dim, int(out_dim * expand_ratio), kernel_size=1,stride=1,padding=0)
        self.f2 = nn.Conv2d(out_dim, int(out_dim * expand_ratio), kernel_size=1,stride=1,padding=0)
        self.act = nn.SiLU(True)
        self.g = nn.Sequential(
            nn.Conv2d(int(out_dim * expand_ratio), out_dim, kernel_size=1),
            nn.SiLU(True),
            nn.BatchNorm2d(out_dim))

        self.gamma = ElementScale(out_dim, init_value=layer_scale_init_value)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_in):

        x = self.norm(x_in)
        x = self.conv(x)
        short_cut = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)

        x = self.act(x1) * x2
        x = self.gamma(self.g(x))
        x = self.drop_path(x)
        return x + short_cut


class HGLRBlock(nn.Module):

    def __init__(self, in_features, norm_layer, mlp_act_layer, channel_first, ssm_act_layer, hglr_type="gl", **kwargs):
        super(HGLRBlock, self).__init__()
        self.in_features = in_features
        abl_mode = kwargs.get("abl_mode", False)
        self.localRefine = kwargs.get('localRefine', False)
        self.vss_branch = kwargs.get('vss_branch', False)
        self.hglr_type = hglr_type
        # print(abl_mode, not abl_mode)
        if not abl_mode:
            self.localRefine = self.vss_branch = True
        if self.localRefine:
            self.gamma2 = ElementScale(in_features, init_value=1e-4)
            self.local_ctx = LocalContextRefine(in_features=in_features)
        if self.vss_branch:
            self.gamma1 = ElementScale(in_features, init_value=1e-4)
            self.global_ctx = nn.Sequential(
                Permute(0, 2, 3, 1),
                VSSBlock(hidden_dim=in_features, drop_path=0.2, norm_layer=norm_layer, channel_first=channel_first,ssm_act_layer=ssm_act_layer,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'],
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type='v3', mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint'],
                ),
                Permute(0, 3, 1, 2),
            )

        self.out_norm = nn.BatchNorm2d(in_features)

    def forward(self, x):
        if not self.localRefine and not self.vss_branch:
            return self.out_norm(x)
        if self.hglr_type == 'gl':
            if self.vss_branch:
                x = x + self.gamma1(self.global_ctx(x))
            if self.localRefine:
                x = x + self.gamma2(self.local_ctx(x))
        elif self.hglr_type == 'lg':
            if self.localRefine:
                x = x + self.gamma2(self.local_ctx(x))
            if self.vss_branch:
                x = x + self.gamma1(self.global_ctx(x))
        elif self.hglr_type == 'weight':
            x1 = self.gamma1(self.global_ctx(x))
            x2 = self.gamma2(self.local_ctx(x))
            weight = torch.sigmoid(x1 + x2)
            x = weight * x1 + (1 - weight) * x2
        elif self.hglr_type == 'sum':
            x1 = self.gamma1(self.global_ctx(x))
            x2 = self.gamma2(self.local_ctx(x))
            x = x + x1 + x2
        else:
            raise NotImplementedError

        return self.out_norm(x)
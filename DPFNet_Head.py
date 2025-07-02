import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import ModuleList, Sequential

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import Upsample
from timm.layers import create_act_layer

from opencd.registry import MODELS
from einops.layers.torch import Rearrange

import warnings
import math


import json
import platform
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

import matplotlib.pyplot as plt
import time

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class CA(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.map = nn.AdaptiveMaxPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )
        self.ma =nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )


    def forward(self, x):
        x_gap = self.gap(x)

        x_map = self.map(x)

        cattn = self.ca(x_gap)

        mattn = self.ma(x_map)

        return cattn ,mattn



class PA(nn.Module):
    def __init__(self, dim):
        super(PA, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        # print(pattn1.shape)
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        # print(x2.shape)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGI(nn.Module):
    def __init__(self, dim, reduction=8): #dim指的是输入tensor的通道数，该模块输入与输出相同
        super(CGI, self).__init__()
        self.sa = SA()
        self.ca = CA(dim, reduction)
        self.pa = PA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,f_low, f_high):

        initial = f_low + f_high
        cattn,mattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn +mattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * f_low + (1 - pattn2) * f_high
        result = self.conv(result)
        return result

class AdaptiveConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False, shrinkage_rate=0.25):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(AdaptiveConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        hidden_dim = int(inc * shrinkage_rate)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inc, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, outc, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        u = self.gate(x)
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype)

        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()

        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()

        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        out = u * out
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)

        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset

class MD(nn.Module):

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='IN'),
                 act_cfg=dict(type='GELU')):
        super(MD, self).__init__()
        self.in_channels = in_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # TODO
        norm_cfg = dict(type='IN')
        act_cfg = dict(type='GELU')

        kernel_size = 5
        self.flow_make = Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      bias=True, groups=in_channels * 2),
            nn.InstanceNorm2d(in_channels * 2),
            nn.GELU(),

        )
        self.flow_makelast = nn.Conv2d(in_channels * 2, 4, kernel_size=1, padding=0, bias=False)
        self.AdaptiveDeformableConvolution = AdaptiveConv2d(in_channels, in_channels)

    def forward(self, x1, x2):
        u1 = self.AdaptiveDeformableConvolution(x1)
        u2 = self.AdaptiveDeformableConvolution(x2)
        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        flow = self.flow_makelast(flow)
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(u1, f1) - u2
        x2_feat = self.warp(u2, f2) - u1

        output = x1_feat + x2_feat
        return output

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_pool_out = self.avg_pool(x)
        max_out_out = self.max_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        max_out = self.fc2(self.relu1(self.fc1(max_out_out)))
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class CA_gai(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(CA_gai, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool_y = nn.AdaptiveAvgPool2d(1)
        self.max_pool_y = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x, y):
        avg_pool_out = self.avg_pool(x)
        max_out_out = self.max_pool(x)
        avg_ax_out = avg_pool_out + max_out_out

        avg_pool_out_y = self.avg_pool_y(y)
        max_out_out_y = self.max_pool_y(y)
        avg_ax_out_y = avg_pool_out_y + max_out_out_y

        avg_out = self.fc2(self.relu1(self.fc1(avg_ax_out + avg_ax_out_y)))
        return self.sigmod(avg_out)

class SA_gai(nn.Module):
    def __init__(self):
        super(SA_gai, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)

        avgout_y = torch.mean(y, dim=1, keepdim=True)
        maxout_y, _y = torch.max(y, dim=1, keepdim=True)
        out_y = torch.cat([avgout_y, maxout_y], dim=1)

        out_out_y = torch.cat([out, out_y],dim=1)

        out_fianl = self.sigmoid(self.conv2d(out_out_y))
        return out_fianl


class ACFF2(nn.Module):

    def __init__(self, channel_L, channel_H):
        super(ACFF2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel_H,out_channels=channel_L,kernel_size=1, stride=1,padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Conv2d(in_channels=2*channel_L, out_channels=channel_L, kernel_size=1, stride=1, padding=0)
        self.BN = nn.BatchNorm2d(channel_L)
        self.relu = nn.ReLU(inplace=True)
        self.ca = CA_gai(in_channels=channel_L,ratio=16)
        self.sa = SA_gai()

    def forward(self, f_low,f_high):

        f_high = self.relu(self.BN(self.conv1(self.up(f_high))))

        adaptive_ca = self.ca(f_low, f_high)
        out_ca = f_low * adaptive_ca + f_high * (1-adaptive_ca) # B,C_l,h,w

        adaptive_sa = self.sa(f_low, f_high)
        out_sa = f_low * (1-adaptive_sa) + f_high * adaptive_sa

        out = out_ca + out_sa

        return out


class CatUP(nn.Module):
    def __init__(self, channel_L, channel_H):
        super(CatUP, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel_H,out_channels=channel_L,kernel_size=1, stride=1,padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.Conv = nn.Sequential(
            nn.Conv2d(channel_L+channel_H,channel_L,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(channel_L),
            nn.ReLU(),
            nn.Conv2d(channel_L, channel_L, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(channel_L),
            )
        self.sigmod = nn.Sigmoid()
        self.ca = ChannelAttention(in_channels=channel_L,ratio=16)

    def forward(self, f_low,f_high):

        f_high =self.up(f_high)

        f_cat = torch.cat((f_low,f_high),dim=1)
        out = self.Conv(f_cat)

        att = self.ca(out)
        output = att*out

        return output

class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d

        self.cbam = CBAM(channel = self.mid_d)

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.mid_d*2, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        d_33 = self.conv3(self.conv2(x))

        d_33_x = self.cbam(self.conv4(torch.cat([d_33,x],dim=1))) + d_33

        return d_33_x

@MODELS.register_module()
class head(BaseDecodeHead):


    def __init__(self,
                 in_channels=256,
                 channels=32,
                 embed_dims=64,
                 enc_depth=1,
                 enc_with_pos=True,
                 dec_depth=8,
                 num_heads=8,
                 drop_rate=0.,
                 pool_size=2,
                 pool_mode='max',
                 use_tokenizer=True,
                 token_len=4,
                 pre_upsample=2,
                 upsample_size=4,
                 interplate_size=(128, 128),
                 final_interplate_size=(256, 256),
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(in_channels, channels, **kwargs)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.embed_dims = embed_dims
        self.use_tokenizer = use_tokenizer
        self.num_heads = num_heads
        self.channels = channels

        self.hidden = channels*3
        self.interplate_size = interplate_size
        self.final_interplate_size = final_interplate_size

        self.sup_channels = in_channels

        self.ACFF3 = ACFF2(channel_L=self.sup_channels[2], channel_H=self.sup_channels[3])
        self.ACFF2 = ACFF2(channel_L=self.sup_channels[1], channel_H=self.sup_channels[2])
        self.ACFF1 = ACFF2(channel_L=self.sup_channels[0], channel_H=self.sup_channels[1])

        self.sam_p4 = SupervisedAttentionModule(self.sup_channels[3])
        self.sam_p3 = SupervisedAttentionModule(self.sup_channels[2])
        self.sam_p2 = SupervisedAttentionModule(self.sup_channels[1])
        self.sam_p1 = SupervisedAttentionModule(self.sup_channels[0])

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upsample8 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upsample_final = nn.Upsample(scale_factor=4, mode='bilinear')


        self.conv4 = nn.Conv2d(self.sup_channels[3], self.sup_channels[0], kernel_size=1)
        self.conv3 = nn.Conv2d(self.sup_channels[2], self.sup_channels[0], kernel_size=1)
        self.conv2 = nn.Conv2d(self.sup_channels[1], self.sup_channels[0], kernel_size=1)

        self.conv_final1 = nn.Conv2d(self.sup_channels[0], self.channels, kernel_size=1)

    def huifuqian(self,inputs):
        d4 = inputs[3]
        d3 = inputs[2]
        d2 = inputs[1]
        d1 = inputs[0]

        p4 = self.sam_p4(d4)

        ACFF_43 = self.ACFF3(d3, p4)
        p3 = self.sam_p3(ACFF_43)

        ACFF_32 = self.ACFF2(d2, p3)
        p2 = self.sam_p2(ACFF_32)

        ACFF_21 = self.ACFF1(d1, p2)
        p1 = self.sam_p1(ACFF_21)

        p4_up = self.upsample8(p4)
        p4_up =self.conv4(p4_up)

        p3_up = self.upsample4(p3)
        p3_up = self.conv3(p3_up)

        p2_up = self.upsample2(p2)
        p2_up = self.conv2(p2_up)

        p= p1+p2_up+p3_up+p4_up

        p_out=self.conv_final1(p)

        out = self.upsample_final(p_out)

        return  out

    def _transform_inputs(self, inputs):

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)

        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs



    def forward(self, inputs):
        new_inputs = self.huifuqian(inputs)
        output = self.cls_seg(new_inputs)


        return output


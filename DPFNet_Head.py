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


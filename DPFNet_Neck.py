import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from timm.models.layers import trunc_normal_
from opencd.registry import MODELS

import matplotlib.pyplot as plt
import time
import numpy as np
import cv2

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class quzaosheng(nn.Module):  #去噪声
    def __init__(self, dim, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 16, bias=False),
            nn.ReLU(),
            nn.Linear(dim // 16, dim, bias=False),
            nn.Sigmoid()
        )
        self.conv1 = nn.Sequential(nn.Conv2d(dim,dim,kernel_size=3,padding=1,stride=1),
                                  nn.BatchNorm2d(dim),
                                  nn.ReLU())

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape

        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)   #复数张量，实部虚部
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x_fft = x.reshape(B, C, H, W)
        return x_fft

class edge_e(nn.Module):
    def __init__(self, in_channel):
        super(edge_e, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.PReLU = nn.PReLU(in_channel)

    def forward(self, x):
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight * x + x
        return out



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_h = a_h.expand(-1,-1,h,w)
        a_w = a_w.expand(-1, -1, h, w)

        # out = identity * a_w * a_h

        return a_w , a_h


class CoDEM2(nn.Module):
    def __init__(self,channel_dim, H, W):
        super(CoDEM2, self).__init__()

        self.channel_dim=channel_dim

        self.Conv3 = nn.Conv2d(in_channels=2*self.channel_dim,out_channels=2*self.channel_dim,kernel_size=3,stride=1,padding=1)
        self.Conv1 = nn.Conv2d(in_channels=2*self.channel_dim,out_channels=self.channel_dim,kernel_size=1,stride=1,padding=0)

        self.BN1 = nn.BatchNorm2d(2*self.channel_dim)
        self.BN2 = nn.BatchNorm2d(self.channel_dim)
        self.ReLU = nn.ReLU(inplace=True)

        self.coAtt_1 = CoordAtt(inp=channel_dim, oup=channel_dim, reduction=16)

        self.quzaosheng = quzaosheng(self.channel_dim,H,W//2+1)
        self.edge_e = edge_e(self.channel_dim)

    def forward(self,x1,x2):   #90_29去掉CA注意力机制
        B,C,H,W = x1.shape
        f_d = torch.abs(x1-x2) #B,C,H,W
        f_c = torch.cat((x1, x2), dim=1)  # B,2C,H,W
        z_c = self.quzaosheng(self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.Conv3(f_c)))))))

        z_d = self.edge_e(f_d)

        out = z_d + z_c

        return out


@MODELS.register_module()
class DPFNet_Neck(BaseModule):

    def __init__(self,
                 in_channels=None,  #[64,128,256,512]
                 channels=None, #[64,128,256,512]
                 out_indices=(0, 1, 2, 3),
                 show_Feature_Maps=True):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.out_indices = out_indices

        self.show_Feature_Maps = show_Feature_Maps

        self.CoDEM2 = nn.ModuleList()
        H = [64,32,16,8]
        W = [64,32,16,8]
        for i in range(len(self.channels)):

            self.CoDEM2.append(CoDEM2(self.in_channels[i],H[i],W[i]))

    def forward(self, x1, x2):

        assert len(x1) == len(x2), "The features x1 and x2 from the" \
            "backbone should be of equal length"

        outs = []

        for i in range(len(x1)):
            out = self.CoDEM2[i](x1[i], x2[i])
            outs.append(out)

        return tuple(outs)
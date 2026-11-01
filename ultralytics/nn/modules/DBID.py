# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""HDFI-Net modules; forward computations retained from the research implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import autopad


class DBIDConv(nn.Module):
    """Keep the research DBID convolution wrapper and its original fusion behavior."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SpatialGuidance(nn.Module):
    """Parameter-free spatial weights obtained by channel averaging."""

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x.mean(dim=1, keepdim=True))


class ChannelGuidance(nn.Module):
    """Parameter-free channel weights obtained by global average pooling."""

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.gap(x))


class DBID(nn.Module):
    """Dual-Branch Interactive Downsampling with mixed pooling and sequential spatial-channel guidance."""

    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = DBIDConv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = DBIDConv(c1 // 2, self.c, 1, 1, 0)
        self.spatial_attn = SpatialGuidance()
        self.channel_attn = ChannelGuidance()

    def forward(self, x):
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2_max = F.max_pool2d(x2, 3, 2, 1)
        x2_avg = F.avg_pool2d(x2, 3, 2, 1)
        x2 = 0.5 * x2_max + 0.5 * x2_avg
        x2 = self.cv2(x2)
        spatial_weight = self.spatial_attn(x2)
        x1 = x1 * spatial_weight
        channel_weight = self.channel_attn(x1)
        x2 = x2 * channel_weight
        return torch.cat([x1, x2], dim=1)


__all__ = ("SpatialGuidance", "ChannelGuidance", "DBID", "DBIDConv")

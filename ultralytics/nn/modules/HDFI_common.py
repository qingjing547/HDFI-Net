# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""HDFI-Net modules; forward computations retained from the research implementation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSRT(nn.Module):
    """Parametric SiLU residual transform: x + scale * SiLU(x) + bias."""

    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x + self.scale * F.silu(x) + self.bias


class ECA(nn.Module):
    """Efficient channel attention used at the GEB output."""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(3, k)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)
        return x * y


class DRConv(nn.Module):
    """Directional reparameterizable convolution with three depthwise training branches."""

    def __init__(self, c1, c2, k=3, s=1, deploy=False):
        super().__init__()
        c1, c2, k, s = (int(c1), int(c2), int(k), int(s))
        self.c1, self.c2, self.k, self.s = (c1, c2, k, s)
        self.deploy = deploy
        p = k // 2
        if deploy:
            self.dw_reparam = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=True)
        else:
            self.dw_3x3 = nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False)
            self.bn_3x3 = nn.BatchNorm2d(c1)
            self.dw_1x3 = nn.Conv2d(c1, c1, (1, 3), 1, (0, 1), groups=c1, bias=False)
            self.bn_1x3 = nn.BatchNorm2d(c1)
            self.dw_3x1 = nn.Conv2d(c1, c1, (3, 1), 1, (1, 0), groups=c1, bias=False)
            self.bn_3x1 = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            dw_out = self.dw_reparam(x)
        else:
            out_3x3 = self.bn_3x3(self.dw_3x3(x))
            out_1x3 = self.bn_1x3(self.dw_1x3(x))
            out_3x1 = self.bn_3x1(self.dw_3x1(x))
            dw_out = out_3x3 + out_1x3 + out_3x1
        return self.act(self.bn_pw(self.pw(dw_out)))

    def _pad_to_3x3(self, kernel):
        h, w = (kernel.shape[2], kernel.shape[3])
        pad_h = (3 - h) // 2
        pad_w = (3 - w) // 2
        return F.pad(kernel, [pad_w, pad_w, pad_h, pad_h])

    def _fuse_conv_bn(self, conv_weight, bn):
        bn_mean = bn.running_mean
        bn_var = bn.running_var
        bn_gamma = bn.weight
        bn_beta = bn.bias
        bn_eps = bn.eps
        std = (bn_var + bn_eps).sqrt()
        t = (bn_gamma / std).view(-1, 1, 1, 1)
        kernel_fused = conv_weight * t
        bias_fused = bn_beta - bn_mean * bn_gamma / std
        return (kernel_fused, bias_fused)

    def reparameterize(self):
        if self.deploy:
            return
        kernel_3x3, bias_3x3 = self._fuse_conv_bn(self.dw_3x3.weight, self.bn_3x3)
        kernel_1x3, bias_1x3 = self._fuse_conv_bn(self.dw_1x3.weight, self.bn_1x3)
        kernel_3x1, bias_3x1 = self._fuse_conv_bn(self.dw_3x1.weight, self.bn_3x1)
        kernel_1x3_padded = self._pad_to_3x3(kernel_1x3)
        kernel_3x1_padded = self._pad_to_3x3(kernel_3x1)
        kernel_fused = kernel_3x3 + kernel_1x3_padded + kernel_3x1_padded
        bias_fused = bias_3x3 + bias_1x3 + bias_3x1
        self.dw_reparam = nn.Conv2d(self.c1, self.c1, self.k, self.s, self.k // 2, groups=self.c1, bias=True)
        self.dw_reparam.weight.data = kernel_fused
        self.dw_reparam.bias.data = bias_fused
        for name in ["dw_3x3", "bn_3x3", "dw_1x3", "bn_1x3", "dw_3x1", "bn_3x1"]:
            if hasattr(self, name):
                delattr(self, name)
        self.deploy = True


class LRG(nn.Module):
    """Learnable residual gate blending original and processed features."""

    def __init__(self, c_mid, c_out):
        super().__init__()
        c_mid = int(c_mid)
        self.gate_conv = nn.Conv2d(c_mid, 1, 1, bias=True)
        nn.init.zeros_(self.gate_conv.bias)
        nn.init.zeros_(self.gate_conv.weight)

    def forward(self, x, mid, out):
        gate = torch.sigmoid(self.gate_conv(mid))
        return gate * out + (1 - gate) * x


@torch.no_grad()
def reparameterize_hdfi(model):
    """Fuse DRConv branches in-place for deployment; call only on an eval-mode model."""
    if model.training:
        raise ValueError("Call model.eval() before deployment reparameterization.")
    for module in list(model.modules()):
        if isinstance(module, DRConv):
            module.reparameterize()
    return model


__all__ = ("PSRT", "ECA", "DRConv", "LRG", "reparameterize_hdfi")

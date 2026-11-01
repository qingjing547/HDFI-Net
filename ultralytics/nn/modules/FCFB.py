# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""HDFI-Net modules; forward computations retained from the research implementation."""

import torch
from torch import nn

from .conv import Conv
from .HDFI_common import PSRT


class FCFBottleneck(nn.Module):
    """Fusion bottleneck with a standard 3x3 convolution and PSRT."""

    def __init__(self, c1, c2, shortcut=False, g=1, e=0.5):
        super().__init__()
        c1, c2, e = (int(c1), int(c2), float(e))
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_, c2, 3, g=1)
        self.fst = PSRT(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.fst(self.cv2(self.cv1(x)))
        return x + out if self.add else out


class FCFB(nn.Module):
    """Fully Connected Fusion Block for neck fusion, including the P3 output."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=False):
        super().__init__()
        c1, c2, n, e = (int(c1), int(c2), int(n), float(e))
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([FCFBottleneck(self.c, self.c, shortcut, g=1, e=0.5) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


__all__ = ("FCFB", "FCFBottleneck")

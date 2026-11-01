# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""HDFI-Net modules; forward computations retained from the research implementation."""

import torch
import torch.nn as nn

from .conv import Conv
from .HDFI_common import DRConv, PSRT


class DLBottleneck(nn.Module):
    """Directional lightweight bottleneck with PSRT and an optional residual connection."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c1, c2, e = (int(c1), int(c2), float(e))
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = DRConv(c_, c2, k=3)
        self.fst = PSRT(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.fst(self.cv2(self.cv1(x)))
        return x + out if self.add else out

    def reparameterize(self):
        self.cv2.reparameterize()


class DLB(nn.Module):
    """Directional Lightweight Block for shallow feature extraction."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.25, g=1, shortcut=True):
        super().__init__()
        c1, c2, n, e = (int(c1), int(c2), int(n), float(e))
        self.c = int(c2 * e)
        self.cv1 = DRConv(c1, 2 * self.c, k=3)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([DLBottleneck(self.c, self.c, shortcut, g=1, e=0.5) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1]) for m in self.m))
        return self.cv2(torch.cat(y, 1))

    def reparameterize(self):
        self.cv1.reparameterize()
        for m in self.m:
            m.reparameterize()


__all__ = ("DLBottleneck", "DLB")

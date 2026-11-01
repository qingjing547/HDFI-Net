# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""HDFI-Net modules; forward computations retained from the research implementation."""

import torch
import torch.nn as nn

from .conv import Conv
from .HDFI_common import DRConv, ECA, LRG, PSRT


class GEBottleneck(nn.Module):
    """Directional bottleneck with PSRT and a learnable residual gate."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c1, c2, e = (int(c1), int(c2), float(e))
        c_ = int(c2 * e)
        self.c_ = c_
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = DRConv(c_, c2, k=3)
        self.fst = PSRT(c2)
        self.add = shortcut and c1 == c2
        if self.add:
            self.gate = LRG(c_, c2)

    def forward(self, x):
        mid = self.cv1(x)
        out = self.fst(self.cv2(mid))
        if self.add:
            return self.gate(x, mid, out)
        return out

    def reparameterize(self):
        self.cv2.reparameterize()


class GEB(nn.Module):
    """Gated Enhancement Block for deep semantic modeling and high-level output."""

    def __init__(self, c1, c2, n=1, c3k=True, e=0.5, g=1, shortcut=True):
        super().__init__()
        c1, c2, n, e = (int(c1), int(c2), int(n), float(e))
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([GEBottleneck(self.c, self.c, shortcut, g=1, e=0.5) for _ in range(n)])
        self.attention = ECA(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1]) for m in self.m))
        out = self.cv2(torch.cat(y, 1))
        return self.attention(out)

    def reparameterize(self):
        for m in self.m:
            m.reparameterize()


__all__ = ("GEBottleneck", "GEB")

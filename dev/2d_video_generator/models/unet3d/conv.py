import torch
import torch.nn as nn
from .swish import Swish

class ConvBlock1D(nn.Module):
    """Convolution blocks of the form specified by `seq`.
    """
    def __init__(self, in_channels, out_channels=None, mid_channels=None,
            kernel_size=3, seq='CBA'):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels is None:
            self.mid_channels = max(in_channels, out_channels)
        self.kernel_size = kernel_size

        self.bn_channels = in_channels
        self.idx_conv = 0
        self.num_conv = sum([seq.count(l) for l in ['U', 'D', 'C']])

        layers = [self._get_layer(l) for l in seq]

        self.convs = nn.Sequential(*layers)

    def _get_layer(self, l):
        if l == 'U':
            in_channels, out_channels = self._setup_conv()
            return nn.ConvTranspose1d(in_channels, out_channels, 2, stride=2)
        elif l == 'D':
            in_channels, out_channels = self._setup_conv()
            return nn.Conv1d(in_channels, out_channels, 2, stride=2)
        elif l == 'C':
            in_channels, out_channels = self._setup_conv()
            return nn.Conv1d(in_channels, out_channels, self.kernel_size)
        elif l == 'B':
            return nn.BatchNorm1d(self.bn_channels)
        elif l == 'A':
            return Swish()
        else:
            raise NotImplementedError('layer type {} not supported'.format(l))

    def _setup_conv(self):
        self.idx_conv += 1

        in_channels = out_channels = self.mid_channels
        if self.idx_conv == 1:
            in_channels = self.in_channels
        if self.idx_conv == self.num_conv:
            out_channels = self.out_channels

        self.bn_channels = out_channels

        return in_channels, out_channels

    def forward(self, x):
        return self.convs(x)

def narrow_like(a, b):
    """Narrow a to be like b.
    Try to be symmetric but cut more on the right for odd difference,
    consistent with the downsampling.
    """
    for dim in range(2, 3):
        width = a.size(dim) - b.size(dim)
        half_width = width // 2
        a = a.narrow(dim, half_width, a.size(dim) - width)
    return a

class ConvBlock3D(nn.Module):
    """Convolution blocks of the form specified by `seq`.
    """
    def __init__(self, in_channels, out_channels=None, mid_channels=None,
            kernel_size=3, seq='CBA'):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_channels is None:
            self.mid_channels = max(in_channels, out_channels)
        self.kernel_size = kernel_size

        self.bn_channels = in_channels
        self.idx_conv = 0
        self.num_conv = sum([seq.count(l) for l in ['U', 'D', 'C']])

        layers = [self._get_layer(l) for l in seq]

        self.convs = nn.Sequential(*layers)

    def _get_layer(self, l):
        if l == 'U':
            in_channels, out_channels = self._setup_conv()
            return nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        elif l == 'D':
            in_channels, out_channels = self._setup_conv()
            return nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif l == 'C':
            in_channels, out_channels = self._setup_conv()
            return nn.Conv3d(in_channels, out_channels, self.kernel_size)
        elif l == 'B':
            return nn.BatchNorm3d(self.bn_channels)
        elif l == 'A':
            return Swish()
        else:
            raise NotImplementedError('layer type {} not supported'.format(l))

    def _setup_conv(self):
        self.idx_conv += 1

        in_channels = out_channels = self.mid_channels
        if self.idx_conv == 1:
            in_channels = self.in_channels
        if self.idx_conv == self.num_conv:
            out_channels = self.out_channels

        self.bn_channels = out_channels

        return in_channels, out_channels

    def forward(self, x):
        return self.convs(x)

def narrow_like(a, b):
    """Narrow a to be like b.
    Try to be symmetric but cut more on the right for odd difference,
    consistent with the downsampling.
    """
    for dim in range(2, 3):
        width = a.size(dim) - b.size(dim)
        half_width = width // 2
        a = a.narrow(dim, half_width, a.size(dim) - width)
    return a

def narrow_like3D(a, b):
    """Narrow a to be like b.
    Try to be symmetric but cut more on the right for odd difference,
    consistent with the downsampling.
    """
    for d in range(2, a.dim()):
        width = a.shape[d] - b.shape[d]
        half_width = width // 2
        a = a.narrow(d, half_width, a.shape[d] - width)
    return a

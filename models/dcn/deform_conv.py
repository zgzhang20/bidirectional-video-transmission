import math
import logging

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from torch.nn.modules.utils import _pair


logger = logging.getLogger('base')


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        *,
        offset_groups=1,
        with_mask=False
    ):
        super().__init__()
        assert in_dim % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.bias = None

        self.with_mask = with_mask
        if with_mask:
            # batch_size, (2+1) * offset_groups * kernel_height * kernel_width, out_height, out_width
            self.param_generator = nn.Conv2d(in_dim, 3 * offset_groups * kernel_size * kernel_size, 3, 1, 1)
        else:
            self.param_generator = nn.Conv2d(in_dim, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.param_generator(x)
            mask = None
        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return x


class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()

        assert not bias
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        return DeformableConv2d(x, offset)


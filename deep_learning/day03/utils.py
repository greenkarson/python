import torch
import math
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


def re_ch(filters, width_coefficient=None):
    if not width_coefficient:
        return filters
    filters *= width_coefficient
    new_filters = max(8, int(filters + 8 / 2) // 8 * 8)
    if new_filters < 0.9 * filters:
        new_filters += 8
    return int(new_filters)


def re_dp(repeats, depth_coefficient=None):
    if not depth_coefficient: return repeats
    return int(math.ceil(depth_coefficient * repeats))


def pad_same(size, kernel_size, stride, dilation):
    o = math.ceil(size / stride)
    pad = max((o - 1) * stride + (kernel_size - 1) * dilation + 1 - size, 0)
    pad_0 = pad // 2
    pad_1 = pad - pad_0
    return pad, pad_0, pad_1


class Conv2dSamePadding(nn.Module):

    def __init__(self, image_size, in_channels, out_channels, kernel_size, stride=1, dilation=1, group=1, bias=True):
        super().__init__()

        h_pad, h_pad_0, h_pad_1 = pad_same(image_size[0], kernel_size, stride, dilation)
        w_pad, w_pad_0, w_pad_1 = pad_same(image_size[1], kernel_size, stride, dilation)

        self.pad = [w_pad_0, w_pad_1, h_pad_0, h_pad_1]

        if h_pad > 0 or w_pad > 0:
            self.static_padding = nn.ZeroPad2d((w_pad_0, w_pad_1, h_pad_0, h_pad_1))
        else:
            self.static_padding = nn.Identity()

        self.module = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                dilation=dilation, groups=group, bias=bias)

    def forward(self, x):
        x = self.static_padding(x)
        return self.module(x)


class Conv2dSamePaddingBNSwish(nn.Module):

    def __init__(self, image_size, in_channels, out_channels, kernel_size, stride=1, group=1, bias=True):
        super().__init__()
        self.sequential = nn.Sequential(
            Conv2dSamePadding(image_size, in_channels, out_channels, kernel_size, stride, group=group, bias=bias),
            nn.BatchNorm2d(out_channels, 1e-3, 1e-2),
            Swish()
        )

    def forward(self, x):
        return self.sequential(x)


class SEModule(nn.Module):

    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=1, padding=0, bias=False),
            Swish(),
            nn.Conv2d(squeeze_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        h = self.se(x)
        return x * torch.sigmoid(h)

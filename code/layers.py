""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

BIAS = False


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=BIAS),
        )

    def forward(self, x):
        return self.single_conv(x)

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvRelu, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=BIAS),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnRelu, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=BIAS),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

class ConvSpRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvSpRelu, self).__init__()
        self.single_conv = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=BIAS)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvRelu(channels, channels),
            Conv(channels, channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Sequential(
                        ConvRelu(in_channels, out_channels),
                        ConvRelu(out_channels, out_channels)
                        )


    def forward(self, x):
        x = self.up(x)
        return self.conv(x)




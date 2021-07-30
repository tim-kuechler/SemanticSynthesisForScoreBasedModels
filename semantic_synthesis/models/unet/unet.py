"""The Unet model. Copied from https://github.com/milesial/Pytorch-UNet"""
import torch

from .layers import *


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.n_channels = config.data.n_channels + (1 if config.model.conditional else 0)
        self.n_classes = config.data.n_labels
        self.bilinear = config.model.bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x, noise=None):
        if noise is not None:
            # Combine x and timesteps
            noise = torch.log(noise)
            noise = torch.unsqueeze(noise, -1)
            noise = torch.unsqueeze(noise, -1)
            noise = noise.expand(noise.shape[0], 1, x.shape[2])
            noise = torch.unsqueeze(noise, -1)
            noise = noise.expand(noise.shape[0], 1, x.shape[2], x.shape[3])
            x = torch.cat([x, noise], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
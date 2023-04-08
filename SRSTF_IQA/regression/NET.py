import os
import math
# import data
import numpy as np
from scipy.io import loadmat
from torchvision import transforms, models
import torch
from torch import nn


class Residual_Block(nn.Module):
    def __init__(self, channal):
        super(Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channal, out_channels=channal, kernel_size=3, stride=1, padding=0, bias=True)
        self.LeakyRelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=channal, out_channels=channal, kernel_size=3, stride=1, padding=0, bias=True)

    def forward(self, x):
        input = x
        output = self.conv1(x)
        output = self.LeakyRelu(output)
        output = self.conv2(output)

        [b, c, m, n] = input.shape
        input_cut4 = input[:, :, 2:m - 2, 2:n - 2]

        output = torch.add(output, input_cut4)

        return output


class H_Net(nn.Module):
    def __init__(self):
        super(H_Net, self).__init__()

        self.cov_share = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),

        )

        self.cov_in = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(),

        )

        self.residual = self.make_layer(Residual_Block, 256, 6)

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(32, affine=True),
            nn.LeakyReLU()
        )

        self.cov_out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(1, affine=True),
            nn.Sigmoid(),
        )

    def make_layer(self, block, channel, num):
        layers = []
        for i in range(num):
            layers.append(block(channel))
        return nn.Sequential(*layers)

    def forward(self, sr_s, bls):
        sr_s = self.cov_share(sr_s)
        bls = self.cov_share(bls)
        # diff = sr_s - bls

        x = torch.cat((sr_s, bls), dim=1)
        x = self.cov_in(x)
        x = self.residual(x)
        feature = self.feature(x)

        out = self.cov_out(feature)
        # feature = torch.flatten(feature)

        return feature, out


class T1_Net(nn.Module):
    def __init__(self):
        super(T1_Net, self).__init__()

        self.cov_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(),

        )

        self.residual = self.make_layer(Residual_Block, 256, 5)

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(32, affine=True),
            nn.LeakyReLU()
        )

        self.cov_out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(1, affine=True),
            nn.Sigmoid(),
        )

    def make_layer(self, block, channel, num):
        layers = []
        for i in range(num):
            layers.append(block(channel))
        return nn.Sequential(*layers)

    def forward(self, sr_t):
        x = self.cov_in(sr_t)
        x = self.residual(x)
        feature = self.feature(x)

        out = self.cov_out(feature)

        # feature = torch.flatten(feature)
        # pred = torch.flatten(x)

        return feature, out

import torch
import numpy as np
# import data
import torch.nn as nn
import math
from torch.autograd import Variable


class Reg_ResidualBlock(nn.Module):
    def __init__(self, in_channels=64):
        super(Reg_ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                               bias=True)

    def forward(self, x):
        identity_data = x
        output = self.conv1(x)
        output = self.relu(self.bn1(output))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output, identity_data)
        return output


class HT_Reg1_256(nn.Module):
    def __init__(self):
        super(HT_Reg1_256, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 1),
            nn.Sigmoid()

        )

    def forward(self, fg):
        prd_score = self.fc(fg)
        return prd_score




class HT_Reg_Spp(nn.Module):
    def __init__(self):
        super(HT_Reg_Spp, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(2772, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, h_feature, h_prd_map, h_weight, t_feature, t_prd_map, t_weight):
        # print(h_feature.shape, h_prd_map.shape, t_feature.shape, t_prd_map.shape)

        h_out = torch.cat([h_feature, h_prd_map], dim=1)
        t_out = torch.cat([t_feature, t_prd_map], dim=1)
        out = torch.cat([h_out, t_out], dim=1)

        num, c, h, w = out.size()

        for i in range(4):  # SPP
            level = i + 1
            if level != 3:
                kernel_size = (math.ceil(h / level), math.ceil(w / level))
                stride = (math.ceil(h / level), math.ceil(w / level))
                pooling = (
                math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

                pooling_layer = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pooling)
                res = pooling_layer(out)

                if i == 0:
                    flatten = res.view(num, -1)
                else:
                    flatten = torch.cat([flatten, res.view(num, -1)], dim=1)

        final = self.fc(flatten)

        return final


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
        # print(output.shape)
        [b, c, m, n] = input.shape
        input_cut4 = input[:, :, 2:m - 2, 2:n - 2]
        # print("int_cut",input_cut4.shape)
        # print(output.shape)
        output = torch.add(output, input_cut4)

        return output


class T_Net(nn.Module):
    def __init__(self):
        super(T_Net, self).__init__()

        self.cov_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),

            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(256, affine=True),
            # nn.ReLU(),

        )

        self.residual = self.make_layer(Residual_Block, 128, 5)

        self.feature = nn.Sequential(
            # nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(128, affine=True),
            # nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.LeakyReLU()
        )

        self.cov_out = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1, affine=True),
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

        feature = torch.flatten(feature)
        # pred = torch.flatten(x)

        return feature, out


class H_Net(nn.Module):
    def __init__(self):
        super(H_Net, self).__init__()

        self.cov_share = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(inplace=True),

        )

        # self.cov_in = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(256, affine=True),
        #     nn.LeakyReLU(),
        #
        # )
        self.cov_in = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(2, affine=True),
            nn.LeakyReLU(),

        )

        self.residual = self.make_layer(Residual_Block, 5)

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.LeakyReLU(),
        )

        self.cov_out = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(3, affine=True),  # Hc的n
            nn.Sigmoid(),
        )

    def make_layer(self, block, num):
        layers = []
        for i in range(num):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, sr_s, bls):
        x = torch.cat((sr_s, bls), dim=1)
        x = self.cov_in(x)
        x = self.cov_share(x)
        # sr_s = self.cov_share(sr_s)
        # bls = self.cov_share(bls)
        # diff = sr_s - bls

        # x = torch.cat((sr_s, bls), dim=1)
        # x = self.cov_in(x)
        x = self.residual(x)
        feature = self.feature(x)

        out = self.cov_out(feature)
        # feature = torch.flatten(feature)

        return feature, out

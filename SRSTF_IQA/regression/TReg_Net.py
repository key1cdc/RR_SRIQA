import torch
# import data
import torch.nn as nn
import math


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


class T_Reg_1_256(nn.Module):
    def __init__(self):
        super(T_Reg_1_256, self).__init__()

        self.residual_1 = self.make_layer(Reg_ResidualBlock, 4, in_channels=64)

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),

        )

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(33, 32, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.ReLU(),
        #
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(693, 256),
            nn.ReLU(True),
            nn.Dropout(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def make_layer(self, block, num_of_layer, in_channels=32):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(in_channels=in_channels))
        return nn.Sequential(*layers)

    def forward(self, feature, pred_map, sr_s, weight):

        out = self.residual_1(feature)
        out = self.conv1(out)
        # print(out.shape, pred_map.shape)

        out = torch.cat([out, pred_map], dim=1)

        out = torch.mul(out, weight)

        num, c, h, w = out.size()

        for i in range(4):  # SPP
            level = i + 1
            if level != 3:
                kernel_size = (math.ceil(h / level), math.ceil(w / level))
                stride = (math.ceil(h / level), math.ceil(w / level))
                pooling = (
                    math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))
                # print(pooling)

                pooling_layer = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pooling)
                res = pooling_layer(out)

                if i == 0:
                    x_flatten = res.view(num, -1)
                else:
                    x_flatten = torch.cat([x_flatten, res.view(num, -1)], dim=1)

        final_256 = self.fc1(x_flatten)
        final_1 = self.fc2(final_256)
        final_1 = torch.unsqueeze(final_1, dim=1)

        return final_256, final_1

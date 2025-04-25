import numpy as np
import torch
import torch.nn as nn
# from .pic2pic_pyconv import CBAMLayer_input
import numpy
import math


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CBAMLayer_input(nn.Module):
    def __init__(self, channel, reduction=32, spatial_kernel=7):
        super(CBAMLayer_input, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, 1, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(1, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        # x = self.conv1_8(x)
        return x


# 对teximage/seg-image 进行两路特征提取
class High_Atten(nn.Module):
    def __init__(self,out):
        super(High_Atten, self).__init__()
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1
                      ))
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1
                      ))
        # self.conv1_4 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2,stride=2
        #               ),nn.BatchNorm2d(64, eps=1e-6, affine=True),nn.PReLU())

        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=out, kernel_size=3, padding=1, stride=1
                      ))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=out, out_channels=out, kernel_size=3, padding=1, stride=1
                      ))
        # self.conv1_8 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=CR, kernel_size=5, padding=2,
        #               ))
        self.relu = nn.PReLU()
        self.CBAM = CBAMLayer_input(out)
        # self.ViT = model_vit()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(192, 192 // 96, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(192 // 96, 192, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv2_0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size=3, padding=1, stride=1
                      ))
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=out, kernel_size=3, padding=1, stride=1
                      ))
        # x2 = model_vit(x)

    def forward(self, x):
        x2 = self.conv2_0(x)
        x_shortcut = x2
        x3 = self.se(x2)
        x2 = x2 * x3
        x2 = x2 + x_shortcut
        x2 = self.relu(x2)

        x_shortcut = x2
        x3 = self.se(x2)
        x3 = x2 * x3
        x3 = x3 + x_shortcut
        x3 = self.relu(x3)
        x3 = self.conv2_1(x3)

        ##################################################
        x1 = self.conv1_0(x)
        x1 = self.relu(x1)

        x1 = self.conv1_1(x1)
        x1 = self.relu(x1)

        x1 = self.conv1_5(x1)
        x1 = self.relu(x1)

        residual = x1
        x1 = self.conv1_6(x1)
        x1 = self.relu(x1)
        x1 = self.conv1_6(x1)
        x1 = self.CBAM(x1)
        x1 = x1 + residual

        x1 = self.relu(x1)

        residual = x1
        x1 = self.conv1_6(x1)
        x1 = self.relu(x1)
        x1 = self.conv1_6(x1)
        x1 = self.CBAM(x1)
        x1 = x1 + residual
        x1 = self.relu(x1)

        return x1, x3


# 对两路提取的特征进行attention和特征复合
# 可多个形成grop
class High_dual(nn.Module):
    def __init__(self, nf1, nf2, ksize1=3, ksize2=3, reduction=4):
        super().__init__()

        self.conv1_1 = nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2)
        self.conv2_1 = nn.Conv2d(2 * nf1, nf1, ksize1, 1, ksize1 // 2)

        self.CA_body1 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(nf1 + nf2, nf1, ksize1, 1, ksize1 // 2),
            CALayer(nf1, reduction))

        self.CA_body2 = CALayer(nf2, reduction)
        self.relu = nn.PReLU()

    def forward(self, x1, x2):

        x1_1 = self.conv1_1(x1)
        # x1_2 = self.conv1_2(x1)
        # x1_3 = self.conv1_3(x1)
        # x_1_sum = x1_1 + x1_2 + x1_3

        x1_1 = self.relu(x1_1)

        x2_1 = self.conv1_1(x2)
        # x2_2 = self.conv1_2(x2)
        # x2_3 = self.conv1_3(x2)
        # x_2_sum = x2_1 + x2_2 + x2_3

        x1_cat = torch.cat((x1_1,x2_1),dim=1)

        f1 = self.conv2_1(x1_cat)
        # x1_2 = self.conv2_2(x1_cat)
        # x1_3 = self.conv2_3(x1_cat)
        # f1 = x1_1 + x1_2 + x1_3


        x2 = self.relu(x2_1)
        f2 = self.conv1_1(x2)
        # x2_2 = self.conv1_2(x2)
        # x2_3 = self.conv1_3(x2)
        # f2 = x2_1 + x2_2 + x2_3


        ca_f1 = self.CA_body1(torch.cat([f1, f2], dim=1))
        ca_f2 = self.CA_body2(f2)

        x1 = x1 + ca_f1
        x2 = x2 + ca_f2
        return x1, x2


# 两路特征合2为1
class dual_one(nn.Module):
    def __init__(self,inchannel, outchannel):
        super(dual_one, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, padding=1, stride=1
                      ))

    def forward(self, x1, x2):
        x1_1 = torch.cat((x1, x2), dim=1)
        x1_1 = self.conv1_1(x1_1)
        return x1_1


class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

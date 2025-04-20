import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
# from models.pytorch_gdn import GDN
from .AT import GDN

# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


# class noise_attention_Module(nn.Module):  # 合成图降维
#     def __init__(self, inchannel):
#         super(noise_attention_Module, self).__init__()
#         self.ave = nn.AdaptiveAvgPool2d(1)
#         self.se = nn.Sequential(
#             nn.Conv2d(inchannel + 1, int(inchannel / 4), kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(int(inchannel / 4), inchannel, kernel_size=1),
#             nn.Sigmoid()
#         )
#         # self.line = nn.Linear(4,1)
#
#         self.conv1_0 = nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=1, stride=1)
#
#         self.relu = nn.PReLU()
#
#     def forward(self, x, y):
#         x_shortcut = x
#         y = SNR_to_noise(y)
#         y = y.tolist()
#         x1 = self.ave(x)
#
#         ba = x1.shape[0]
#         y = torch.tensor(y).to(device)
#         y = y.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
#         y = y.repeat(ba, 1, 1, 1)
#         x1 = torch.cat((y, x1), dim=1)
#         # x1 = x1.type(torch.FloatTensor)
#         x1 = self.se(x1)
#
#         x2 = x * x1
#         x2 = x2 + x_shortcut
#         resi = x2
#         x2 = self.conv1_0(x2)
#         x2 = self.relu(x2)
#         x2 = self.conv1_0(x2)
#         x2 = self.relu(x2)
#         x2 = x2 + resi
#         return x2

class noise_attention_Module(nn.Module):  # 合成图降维
    def __init__(self, inchannel):
        super(noise_attention_Module, self).__init__()
        self.ave = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(inchannel + 1, int(inchannel / 16)),
            nn.PReLU(),
            nn.Linear(int(inchannel / 16), inchannel),
        )

        self.sig = nn.Sigmoid()
        # self.line = nn.Linear(4,1)

        self.conv1_0 = nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, padding=1, stride=1)

        self.relu = nn.PReLU()

    def forward(self, x, y):

        y = torch.tensor(y).unsqueeze(0).unsqueeze(0).to(device)
        y = y.repeat(x.shape[0], 1).to(torch.float32)

        x1 = self.ave(x).view(x.shape[0], -1)
        x2 = self.max(x).view(x.shape[0], -1)
        x1 = torch.cat((y, x1), dim=1)
        x2 = torch.cat((y, x2), dim=1)

        x1 = self.se(x1)
        x2 = self.se(x2)
        x_weight = self.sig(x1 + x2).view(x.shape[0], -1, 1,1)

        x2 = x*x_weight
        resi = x2
        x2 = self.conv1_0(x2)
        x2 = self.relu(x2)
        x2 = self.conv1_0(x2)
        x2 = self.relu(x2)
        x2 = x2 + resi
        return x2

class feature_choice_in(nn.Module):
    def __init__(self):
        super(feature_choice_in, self).__init__()

        self.gdn1 = GDN(128)

        self.rbs = nn.Sequential(  # 蓝1
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
        )
        self.rbss = nn.Sequential(  # 蓝1
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
        )
        self.rbs2 = nn.Sequential(  # 蓝2
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),

        )

        self.rbs3 = nn.Sequential(  # 黄
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.PReLU()
        )
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2),
                                   nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
                                   nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),
                                   nn.PReLU())
        # self.NA_Module1 = noise_attention_Module(64)  # 绿
        self.NA_Module2 = noise_attention_Module(128)

        self.relu = nn.PReLU()
        # self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.ATM = ATM()
        self.CBAM = CBAM(64)
        self.Refine = Refine(256)

    def forward(self, f1, f2, n_var):

        x1 = self.rbs(f1)
        x1 = self.gdn1(x1)
        x1_1 = self.NA_Module2(x1, n_var)
        x2 = self.rbss(f2)
        x2 = self.gdn1(x2)
        x2_1 = self.NA_Module2(x2, n_var)

        x1 = self.rbs2(x1_1)
        x1 = self.gdn1(x1)
        x1_2 = self.NA_Module2(x1, n_var)
        x2 = self.rbs2(x2_1)
        x2 = self.gdn1(x2)
        x2_2 = self.NA_Module2(x2, n_var)

        x1 = self.rbs2(x1_2)
        x1 = self.gdn1(x1)
        x1_3 = self.NA_Module2(x1, n_var)
        x2 = self.rbs2(x2_2)
        x2 = self.gdn1(x2)
        x2_3 = self.NA_Module2(x2, n_var)

        x1 = self.rbs3(x1_3)
        x1 = self.gdn1(x1)
        x1_4 = self.NA_Module2(x1, n_var)
        x2 = self.rbs3(x2_3)
        x2 = self.gdn1(x2)
        x2_4 = self.NA_Module2(x2, n_var)

        t1 = self.ATM(x1_1, x2_1)
        t2 = self.ATM(x1_2, x2_2)
        t3 = self.ATM(x1_3, x2_3)
        t4 = self.ATM(x1_4, x2_4)

        t1 = self.conv1(t1)
        t1 = self.conv2(t1)
        t1 = self.CBAM(t1)

        t2 = self.conv1(t2)
        t2 = self.CBAM(t2)

        t3 = self.conv3(t3)
        t3 = self.CBAM(t3)

        t4 = self.conv3(t4)
        t4 = self.CBAM(t4)

        tt = torch.cat((t1, t2, t3, t4), dim=1)
        # print(tt.shape)
        tt = self.Refine(tt)

        return tt, x2_4


class ResBlock_LeakyReLU_0_Point_1(nn.Module):
    def __init__(self, d_model):
        super(ResBlock_LeakyReLU_0_Point_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(d_model, d_model, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = x + self.conv(x)
        return x


class mix(nn.Module):
    def __init__(self):
        super(mix, self).__init__()
        self.contextualEncoder = nn.Sequential(
            nn.Conv2d(256 + 128, 256, 5, stride=1, padding=2),
            GDN(256),
            ResBlock_LeakyReLU_0_Point_1(256),
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            GDN(256),
            ResBlock_LeakyReLU_0_Point_1(256),
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            GDN(256),
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
        )

    def forward(self, x, y):
        m = torch.cat((x, y), dim=1)
        m = self.contextualEncoder(m)

        return m


class ATM(nn.Module):  # attention transition module
    def __init__(self, ):
        super(ATM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1),
                                   nn.PReLU())
        self.softmax = nn.Softmax(dim=1)

        # self.rbs = nn.Sequential(  # 蓝1
        #     nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1, stride=2),
        #     nn.PReLU(),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
        # )

    def forward(self, x, y):
        x1 = self.conv1(x)
        y1 = self.conv1(y)

        A = self.softmax(x1)
        B = self.softmax(y1)

        x1 = x1 * A

        y1 = y1 * B

        C = torch.cat((x1, y1), dim=1)
        C = self.conv2(C)

        return C


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class Refine(nn.Module):
    def __init__(self, planes, scale_factor=2):
        super(Refine, self).__init__()
        # self.convFS1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.convFS2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

    def forward(self, f):
        sr = self.convFS2(F.relu(f))
        sr = self.convFS3(F.relu(sr))
        s = f + sr
        m = s + F.interpolate(s, size=s.shape[2:4], mode='bilinear')

        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        return m


class feature_choice_out(nn.Module):
    def __init__(self):
        super(feature_choice_out, self).__init__()

        self.DFF = DFF()

    def forward(self, top, bot):

        # p0 = bot[1].unsqueeze(0)
        # p7 = top[-1].unsqueeze(0)
        #
        # x = top[:-1]
        # y = bot[1:]
        out = self.DFF(top, bot)

        # out = torch.cat((p0, p16, p7), dim=0)

        return out


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

        x1_1 = self.relu(x1_1)

        x2_1 = self.conv1_1(x2)

        x1_cat = torch.cat((x1_1, x2_1), dim=1)

        f1 = self.conv2_1(x1_cat)

        x2 = self.relu(x2_1)
        f2 = self.conv1_1(x2)

        ca_f1 = self.CA_body1(torch.cat([f1, f2], dim=1))
        ca_f2 = self.CA_body2(f2)

        x1 = x1 + ca_f1
        x2 = x2 + ca_f2
        return x1, x2


# 两路特征合2为1
class dual_one(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(dual_one, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, padding=1, stride=1),
            nn.PReLU())

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


class DFF(nn.Module):  # 合成图降维
    def __init__(self):
        super(DFF, self).__init__()
        # self.High_Atten = High_semantic_atten.High_Atten(textseg)
        self.High_dual = High_dual(256, 256)
        self.dual_one = dual_one(256 + 256, 256)

    def forward(self, x, y):
        for i in range(3):
            x, y = self.High_dual(x, y)

        out = self.dual_one(x, y)

        return out

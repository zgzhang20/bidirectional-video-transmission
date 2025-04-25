### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import numpy
import math
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# from models.pytorch_gdn import GDN
# from .gmflow import GMFlow
import argparse
import models.High_semantic_atten as High_semantic_atten
from functools import reduce
import torch.nn.functional as F
from torchvision import models
from .key import test_pyconv
from .fusion import feature_choice_in, noise_attention_Module, feature_choice_out, mix
from .AT import AT, AT_weight, SAF_Module, GDN
from .alignnet import PCD_Align, FeatureEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class mi(nn.Module):
    def __init__(self):
        super(mi, self).__init__()

        self.recons_mid_forw = recons_mid_forw()
        self.aligner = PCD_Align()
        self.FeatureEncoder = FeatureEncoder()
        self.AT = AT()
        self.AT_weight = AT_weight(64)
        self.perfect_channel = perfect_channel()
        self.mix = mix()
        self.PF_module = PF_module()
        self.information_de=information_de()



class AF_Module(nn.Module):  # 合成图降维
    def __init__(self, channel):
        super(AF_Module, self).__init__()
        self.ave = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, int(channel / 2), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(channel / 2), out_channels=int(channel / 4), kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(int(channel / 4), channel, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1_0(x)
        residual = x
        x1 = self.ave(x)
        # print(x.shape)torch.Size([8, 4, 1, 1])
        x2 = self.se(x1)
        # print(x2.shape)torch.Size([8, 4, 1, 1])
        x2 = x2 * x
        x2 = x2 + residual
        return x2


class PF_module(nn.Module):  # 原图降维
    def __init__(self, N=192, M=192, side_input_channels=3, num_slices=8):
        super(PF_module, self).__init__()
        self.num_slices = num_slices
        self.conv1_0 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.conv1_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.conv1_2 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.conv1_3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1)

        self.conv1_4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)

        self.scale = nn.Sequential(nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                                   nn.PReLU(),
                                   nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                                   nn.PReLU(),
                                   nn.Conv2d(32, 1, stride=1, kernel_size=3, padding=1),
                                   nn.Sigmoid(),
                                   )

        self.Prelu = nn.ReLU()

        self.hyper_xa = nn.Sequential(
            nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
            )
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, 32, stride=1, kernel_size=3, padding=1),
            )
        )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1),
            )
        )

        self.AF_Module = noise_attention_Module(64)

    def forward(self, es_f, es_b, res, pred, n_var):
        global kk

        es = self.Prelu(self.conv1_1(self.Prelu(self.conv1_1(torch.cat((es_f, es_b), dim=1))).view(3, -1, es_f.shape[2], es_f.shape[3])))

        res = self.Prelu(self.conv1_0(res))
        pred = self.Prelu(self.conv1_0(pred))

        fea = self.Prelu(self.conv1_2(torch.cat((es, res, pred), dim=1)))

        residual = fea
        fea = self.conv1_0(fea)
        fea = self.Prelu(fea)
        fea = self.conv1_0(fea)
        fea = residual + fea
        fea = self.Prelu(fea)
        fea = self.AF_Module(fea, n_var)
        fea_out = self.conv1_3(fea)
        fea_out = self.Prelu(fea_out)

        fea_entropy = self.hyper_xa(fea)

        latent_means, latent_scales = fea_entropy.chunk(2, 1)

        y_slices = fea.chunk(2, 1)

        y_hat_slices = []

        y_shape = fea.shape[2:]
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices[:slice_index])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            # print("slice_index:", slice_index, y_slice.size(), mean_support.size(), mu.size())
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            # print("mu:", mu.size())

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            y_slice = self.cc_scale_transforms[slice_index](scale_support)
            # y_likelihood.append(y_slice_likelihood)
            y_hat_slice = (y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)

            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = self.conv1_4(lrp)

            lrp = 0.5 * torch.tanh(lrp)

            kk = lrp + y_hat_slice

        fea_out = self.scale(kk) * fea_out

        return fea_out


class information_de(nn.Module):  # 原图降维
    def __init__(self, ):
        super(information_de, self).__init__()
        self.conv1_0 =nn.Sequential(nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, padding=0, stride=1),
                                     nn.PReLU(),

                                     )
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=192, kernel_size=1, padding=0, stride=1),
                                     nn.PReLU(),
                                     nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, padding=0, stride=1),
                                     nn.PReLU(),
                                     )


    def forward(self, fram_send, key_fea, n_var):
        fram_send = self.conv1_1(fram_send)

        chun = torch.chunk(fram_send, 3, dim=1)
        # print(es[0].shape)
        es = torch.chunk(chun[0], 2, dim=1)
        es_f = self.conv1_0(es[0]).view(6,-1,fram_send.shape[2],fram_send.shape[3])

        es_b = self.conv1_0(es[1]).view(6,-1,fram_send.shape[2],fram_send.shape[3])
        res = chun[1]
        pred = chun[2]

        return es_f, es_b, res, pred

class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out = self.Conv_BN_ReLU_2(x)
        out_2 = self.downsample(out)
        return out, out_2


class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        # print(x_out.shape)
        # print(out.shape)
        cat_out = torch.cat((x_out, out), dim=1)
        return cat_out
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels = [2 ** (i + 6) for i in range(5)]  # [64, 128, 256, 512, 1024]
        # 下采样
        # print(out_channels)
        # self.d1=DownsampleLayer(3,out_channels[0])#3-64
        # self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
        self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
        # 上采样
        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
        self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
        self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
        # self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        # 输出
        self.o1 = nn.Sequential(
            nn.Conv2d(out_channels[3], out_channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU())
        self.o2 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU())
        self.o3 = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[1], 3, 1, 1),
            #nn.Sigmoid(),
            nn.ReLU())

        self.AF_Module1 = noise_attention_Module(256)
        self.AF_Module2 = noise_attention_Module(128)
        # self.AF_Module1 = AF_Module(32)

    def forward(self, x, n_var):
        # out_1,out1=self.d1(x)
        # out_2,out2=self.d2(out1)
        out_3, out3 = self.d3(x)
        out_4, out4 = self.d4(out3)
        out5 = self.u1(out4, out_4)
        out6 = self.u2(out5, out_3)
        # out7=self.u3(out6,out_2)
        # out8=self.u4(out7,out_1)
        out = self.o1(out6)
        out = self.AF_Module1(out, n_var)

        out = self.o2(out)
        out = self.AF_Module2(out, n_var)

        out = self.o3(out)
        # print(out.shape)
        return out

class recons_mid_forw(nn.Module):  # 原图降维
    def __init__(self, ):
        super(recons_mid_forw, self).__init__()
        self.igdn1 = GDN(128, inverse=True)
        self.igdn3 = GDN(256, inverse=True)
        self.igdn5 = GDN(64, inverse=True)

        self.conv1_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,
                      ))

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,
                      ))

        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1,
                      ))
        self.conv1_3_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1,
                      ))

        self.upsample1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.upsample3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1, stride=1
                      ))
        self.relu = nn.PReLU()
        self.UNet = UNet()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        # self.AF_Module = noise_attention_Module(128)
        self.AF_Module1 = noise_attention_Module(128)
        self.AF_Module2 = noise_attention_Module(256)
        self.SAF1 = SAF_Module(128)
        # self.SAF2 = SAF_Module(256)
        # self.UNet = UNet()
        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.convcc = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)

    def forward(self, recons_input_feature, pred, n_var):

        recons_input_feature = self.relu(self.conv1_1(recons_input_feature))
        pred = self.relu(self.conv1_1(pred))

        # a1 = torch.cat((key_fea[0].unsqueeze(0), fram_send[0].unsqueeze(0), key_fea[1].unsqueeze(0)), dim=1)
        # a2 = torch.cat((key_fea[1].unsqueeze(0), fram_send[1].unsqueeze(0), key_fea[2].unsqueeze(0)), dim=1)
        # a3 = torch.cat((key_fea[2].unsqueeze(0), fram_send[2].unsqueeze(0), key_fea[3].unsqueeze(0)), dim=1)

        c = torch.cat((recons_input_feature, pred), dim=1)

        d = self.relu(self.convcc(c))
        x = self.conv1_0(d)
        x = self.igdn1(x)
        x = self.relu(x)
        x = self.AF_Module1(x, n_var)
        x = self.UNet(x, n_var)

        residule = x
        x = self.conv1_3_3(x)
        x = self.igdn1(x)
        x = self.relu(x)
        x = x + residule
        x = self.AF_Module1(x,n_var)
        x = self.conv1_3_3(x)
        x = self.igdn1(x)
        x = self.relu(x)
        x = self.SAF1(x)

        residule = x
        x = self.conv1_3_3(x)
        x = self.igdn1(x)
        x = self.relu(x)
        x = x + residule
        x = self.AF_Module1(x,n_var)
        x = self.conv1_3_3(x)
        x = self.igdn1(x)
        x = self.relu(x)
        x = self.SAF1(x)


        residule = x
        x = self.conv1_3_3(x)
        x = self.igdn1(x)
        x = self.relu(x)
        x = x + residule
        x = self.AF_Module1(x,n_var)
        x = self.conv1_3_3(x)
        x = self.igdn1(x)
        x = self.relu(x)
        x = self.SAF1(x)

        x = self.conv1_3(x)
        x = self.igdn3(x)
        x = self.relu(x)

        x = self.upsample1(x)

        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.conv1_7(x)
        x = self.sigmoid(x)
        return x


class perfect_channel(nn.Module):  # 原图降维
    def __init__(self):
        super(perfect_channel, self).__init__()
        self.gdn1 = GDN(64)
        self.igdn1 = GDN(64, inverse=True)
        # self.gdn2 = GDN(24, device)
        # self.gdn2 = GDN(CR, device)

        self.rbs = nn.Sequential(  # 蓝1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
        )

        self.rbs2 = nn.Sequential(  # 蓝2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),

        )
        self.rbs3 = nn.Sequential(  # 蓝3
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, stride=1),

        )
        self.trb3 = nn.Sequential(  # 黄
            nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU()
        )

        self.trb2 = nn.Sequential(  # 黄
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU()
        )

        self.trb1 = nn.Sequential(  # 黄
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )

        self.at = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
        )
        self.tat = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x1 = self.rbs(x)
        x1 = self.gdn1(x1)
        r = x1
        x1 = self.at(x1)
        x1 = x1 + r

        x1 = self.rbs2(x1)
        x1 = self.gdn1(x1)
        r = x1
        x1 = self.at(x1)
        x1 = x1 + r

        x1 = self.rbs2(x1)
        x1 = self.gdn1(x1)
        r = x1
        x1 = self.at(x1)
        x1 = x1 + r
        x1 = self.rbs3(x1)
        # print(x1.shape)
        ##############################
        x1 = self.trb3(x1)
        x1 = self.igdn1(x1)
        r = x1
        x1 = self.tat(x1)
        x1 = x1 + r

        x1 = self.trb2(x1)
        x1 = self.igdn1(x1)
        r = x1
        x1 = self.tat(x1)
        x1 = x1 + r

        x1 = self.trb1(x1)

        return x1



def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)

    return x


class Channels():

    def AWGN(self, Tx_sig, snr):
        # print(Tx_sig.shape)
        # Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        # return Rx_sig
        # np.random.seed(seed)  # 设置随机种子
        shpae = Tx_sig.shape
        Tx_sig = torch.flatten(Tx_sig)

        # print(shpae)
        snr = 10 ** (snr / 10.0)
        # print(Tx_sig.shape)
        xpower = (torch.sum(Tx_sig ** 2) / len(Tx_sig)).to(device)
        npower = xpower / snr
        noise = (torch.randn(len(Tx_sig)).to(device)) * (torch.sqrt(npower).to(device))
        # print(torch.randn(len(Tx_sig)).to(device))
        Rx_sig = Tx_sig + noise
        Rx_sig = Rx_sig.reshape(shpae)
        # print(Rx_sig.shape)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        # Rx_sig = Tx_sig
        Rx_sig = self.AWGN(Tx_sig, n_var / 10)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = Tx_sig
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig


def psnr(img1, img2):
    # img1 = np.float64(img1)
    # img2 = np.float64(img2)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def train_pyconv(model, kye_model, opt, bimage, channel, n_var):
    # dot_val = 0.0
    model.train()
    # print(n_var.device)  # 输出：cpu
    opt.zero_grad()
    # a_norm = 0.0
    key, ssim_value, psn, mse, key_send = test_pyconv(kye_model, bimage, channel, n_var)

    for i in range(key.shape[0]):
        bimage[i * 2] = key[i]

    channels = Channels()

    top_image = bimage[:-1]
    bottom_image = bimage[1:]
    predict_feature = bimage[[2 * i - 1 for i in range(1, 4)]]

    top_image_F = model.FeatureEncoder(top_image, n_var)
    bottom_image_F = model.FeatureEncoder(bottom_image, n_var)
    key_feature = model.FeatureEncoder(key, n_var)
    input_feature = model.FeatureEncoder(predict_feature, n_var)
    # print(top_image.shape)
    # print(top_image_F.shape)
    F_offset = model.aligner.MotionEstimation(top_image_F, bottom_image_F)
    B_offset = model.aligner.MotionEstimation(bottom_image_F, top_image_F)

    print(F_offset==B_offset)
    predic_feature = model.aligner.MotionCompensation(F_offset, B_offset, key_feature)
    # print(predic_feature.shape)
    res_feature = input_feature - predic_feature
    feature_choice_out = model.PF_module(F_offset, B_offset, res_feature, input_feature, n_var)

    # x_chn_weight = model.AT(predic_feature)
    #
    # feature_choice_out = model.AT_weight(predic_feature, x_chn_weight)

    fram_send = PowerNormalize(feature_choice_out)  # torch.Size([8, 3, 32, 32])

    if channel == 'AWGN':
        fram_send = channels.AWGN(fram_send, n_var)
        # bid_flow = channels.AWGN(bid_flow, n_var)
    elif channel == 'Rayleigh':
        fram_send = channels.Rayleigh(fram_send, n_var)
        # bid_flow = channels.Rayleigh(bid_flow, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh")
    es_f, es_b, res, pred = model.information_de(fram_send, key_feature, n_var)

    predic_feature_decoder = model.aligner.MotionCompensation(es_f, es_b, key_feature)

    recons_input_feature=res+predic_feature_decoder

    pre_frame = bimage[[2 * i - 1 for i in range(1, 4)]]
    mid = model.recons_mid_forw(recons_input_feature, pred, n_var)
    loss_mid=model.perfect_channel(predict_feature)

    loss = 1 - 0.9*psnr(mid * 255, pre_frame * 255)- 0.1*psnr(loss_mid * 255, pre_frame * 255)

    ssim_value = loss.data.item()
    loss.backward()
    opt.step()

    return ssim_value


def mse(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return mse



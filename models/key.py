### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

import numpy as np
import torch
import torch.nn as nn
import math
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from models.pytorch_gdn import GDN


# device = torch.device("cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class mi(nn.Module):
    def __init__(self):
        super(mi, self).__init__()
        self.key_encoder = key_encoder()
        self.key_decoder = key_decoder()
        self.perfect_channel = perfect_channel()


class key_encoder(nn.Module):  # 原图降维
    def __init__(self, ):
        super(key_encoder, self).__init__()
        self.gdn1 = GDN(256, device)
        #self.gdn2 = GDN(256, device)
        #self.gdn3 = GDN(256, device)
        #self.gdn4 = GDN(256, device)
        self.gdn5 = GDN(1, device)
        # self.gdn2 = GDN(CR, device)
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1, stride=2
                      ))
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2, stride=2
                      ))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2, stride=1
                      ))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2, stride=1
                      ))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=5, padding=2, stride=1
                      ))

        self.relu = nn.PReLU()

        self.AF_Module1 = AF_Module(256)
        #self.AF_Module2 = AF_Module(256)
        #self.AF_Module3 = AF_Module(256)
        #self.AF_Module4 = AF_Module(256)

    def forward(self, x, n_var):
        x1 = self.conv1_0(x)
        x1 = self.gdn1(x1)
        x1 = self.relu(x1)
        x1 = self.AF_Module1(x1, n_var)

        x1 = self.conv1_1(x1)
        x1 = self.gdn1(x1)
        x1 = self.relu(x1)
        x1 = self.AF_Module1(x1, n_var)

        x1 = self.conv1_2(x1)
        x1 = self.gdn1(x1)
        x1 = self.relu(x1)
        x1 = self.AF_Module1(x1, n_var)

        x1 = self.conv1_3(x1)
        x1 = self.gdn1(x1)
        x1 = self.relu(x1)
        x1 = self.AF_Module1(x1, n_var)
        
        residual=x1
        x1 = self.conv1_3(x1)
        x1 = self.gdn1(x1)
        x1 = self.relu(x1)
        x1 = self.AF_Module1(x1+residual, n_var)
        residual=x1
        x1 = self.conv1_3(x1)
        x1 = self.gdn1(x1)
        x1 = self.relu(x1)
        x1 = self.AF_Module1(x1+residual, n_var)

        x1 = self.conv1_4(x1)
        x1 = self.gdn5(x1)

        return x1


class key_decoder(nn.Module):  # 原图降维
    def __init__(self, ):
        super(key_decoder, self).__init__()
        #self.gdn1 = GDN(256, device)
        # self.igdn2 = GDN(3, device, inverse=True)
        # self.igdn3 = GDN(pixel, device, inverse=True)

        self.igdn1 = GDN(32, device, inverse=True)
        #self.igdn2 = GDN(64, device, inverse=True)
       # self.igdn3 = GDN(128, device, inverse=True)
        self.igdn4 = GDN(256, device, inverse=True)
        # self.igdn3 = GDN(pixel, device, inverse=True)
        self.con = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv1_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, padding=2,
                               ))
        self.conv1_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=5, padding=2,
                               ))

        self.conv1_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=2,
                               ))
        self.conv1_2_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=2,
                               ))

        self.conv1_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, padding=2, stride=2, output_padding=1
                               ))

        self.conv1_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=5, padding=2, stride=2, output_padding=1
                               ))

        self.relu = nn.PReLU()

        self.sigmoid = nn.Sigmoid()
        self.AF_Module1 = AF_Module(32)
        self.AF_Module2 = AF_Module(256)
        #self.AF_Module3 = AF_Module(256)
        #self.AF_Module4 = AF_Module(256)

        self.weight_related = weight_related(32)

    def forward(self, x, n_var):
        x = self.con(x)

        x0 = x[0].unsqueeze(0)
        x1 = x[1].unsqueeze(0)
        x2 = x[2].unsqueeze(0)
        x3 = x[3].unsqueeze(0)
        
        x00 = self.weight_related(x1) * x0 + x0
        x11 = self.weight_related(x0) * x1 + x1
        x22 = self.weight_related(x1) * x2 + x2
        x33 = self.weight_related(x2) * x3 + x3

        x = torch.cat((x00, x11, x22, x33), dim=0)

        x = self.conv1_0(x)
        x = self.igdn1(x)
        x = self.relu(x)
        x = self.AF_Module1(x, n_var)

        x = self.conv1_1(x)
        x = self.igdn4(x)
        x = self.relu(x)
        x = self.AF_Module2(x, n_var)

        x = self.conv1_2(x)
        x = self.igdn4(x)
        x = self.relu(x)
        # x = self.AF_Module3(x, n_var)
        x = self.conv1_3(x)
        x = self.igdn4(x)
        x = self.relu(x)

        residual = x
        x = self.conv1_2_2(x)
        x = self.igdn4(x)
        x = self.relu(x)
        x = residual + x
        residual = x
        x = self.conv1_2_2(x)
        x = self.igdn4(x)
        x = self.relu(x)
        x = residual + x

        residual = x
        x = self.conv1_2_2(x)
        x = self.igdn4(x)
        x = self.relu(x)
        x = residual + x

        # x = self.AF_Module4(x, n_var)

        x = self.conv1_4(x)
        # x = self.igdn2(x)
        x = self.sigmoid(x)

        return x


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


class weight_related(nn.Module):  # 合成图降维
    def __init__(self, inchannel):
        super(weight_related, self).__init__()
        self.weg = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(inchannel, inchannel, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(inchannel, inchannel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.weg(x)
        return x


class AF_Module(nn.Module):  # 合成图降维
    def __init__(self, inchannel):
        super(AF_Module, self).__init__()
        self.ave = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(inchannel + 1, int(inchannel / 8), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(inchannel / 8), inchannel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        y = SNR_to_noise(y)
        y = y.tolist()
        x1 = self.ave(x)
        ba = x1.shape[0]
        y = torch.tensor(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        y = y.repeat(ba, 1, 1, 1)
        x1 = torch.cat((y, x1), dim=1)
        x1 = self.se(x1)
        x2 = x * x1

        return x2


class SAF_Module(nn.Module):  # 合成图降维
    def __init__(self, channel):
        super(SAF_Module, self).__init__()
        self.ave = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, int(channel / 2), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(channel / 2), out_channels=int(channel), kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x1 = x + self.se(x)
        x1 = x1 + self.se(x1)
        x1 = x1 + self.se(x1)
        x2 = x + self.se(x)
        x2 = x2 + self.se(x2)
        x2 = x2 + self.se(x2)
        x2 = self.conv1_0(x2)
        x2 = self.sig(x2)
        x2 = x2 * x1
        x2 = x2 + residual
        return x2


class perfect_channel(nn.Module):  # 原图降维
    def __init__(self):
        super(perfect_channel, self).__init__()
        self.gdn1 = GDN(64, device)
        self.igdn1 = GDN(64, device, inverse=True)
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


class Encoder_dense(nn.Module):  # 原图降维
    def __init__(self):
        super(Encoder_dense, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(256, 256))

    def forward(self, x):
        x1 = self.linear(x)
        return x1


def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)

    return x


class Channels():

    def AWGN(self, Tx_sig, snr):
        snr = 10 ** (snr / 10)
        noise_std = 1 / np.sqrt(2 * snr)
        Rx_sig = Tx_sig + torch.normal(0, noise_std, size=Tx_sig.shape).to(device)
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


def train_pyconv(model, opt, bimage, channel, n_var):
    # dot_val = 0.0
    # a_norm = 0.0
    model.train()
    # print(n_var.device)  # 输出：cpu
    opt.zero_grad()
    channels = Channels()
    image = bimage[[2 * i - 2 for i in range(1, 5)]]
    key_frame = model.key_encoder(image, n_var)
    potent = model.perfect_channel(image)

    fram_send = PowerNormalize(key_frame)  # torch.Size([8, 3, 32, 32])
    if channel == 'AWGN':
        fram_send = channels.AWGN(fram_send, n_var)
        # bid_flow = channels.AWGN(bid_flow, n_var)
    elif channel == 'Rayleigh':
        fram_send = channels.Rayleigh(fram_send, n_var)
        # bid_flow = channels.Rayleigh(bid_flow, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh")

    mid = model.key_decoder(fram_send, n_var)

    #loss1 = torch.mean(torch.square(mid * 255 - image * 255))
    #loss2 = torch.mean(torch.square(potent * 255 - image * 255))
    
    loss1 = psnr(image * 255, mid * 255)
    loss2 = psnr(image * 255, potent * 255)
    
    loss = 100-(loss1 + 0.01 * loss2)
    ssim_value = loss.data.item()
    loss.backward()
    opt.step()

    return loss1


def mse(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return mse


def test_pyconv(model, bimage, channel, n_var):
    model.eval()
    # print(n_va r.device)  # 输出：cpu
    # opt.zero_grad()
    channels = Channels()

    image = bimage[[2 * i - 2 for i in range(1, 5)]]
    key_frame = model.key_encoder(image, n_var)
    potent = model.perfect_channel(image)

    fram_send = PowerNormalize(key_frame)  # torch.Size([8, 3, 32, 32])
    if channel == 'AWGN':
        fram_send = channels.AWGN(fram_send, n_var)
        # bid_flow = channels.AWGN(bid_flow, n_var)
    elif channel == 'Rayleigh':
        fram_send = channels.Rayleigh(fram_send, n_var)
        # bid_flow = channels.Rayleigh(bid_flow, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh")

    mid = model.key_decoder(fram_send, n_var)

    mse = torch.mean(torch.square(image * 255 - mid * 255))
    ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)
    loss = ms_ssim_module(mid * 255, image * 255)
    ssim_value = loss.data.item()
    psn = psnr(image * 255, mid * 255)

    return mid, ssim_value, psn, mse, fram_send

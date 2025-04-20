import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Function
from scipy import ndimage
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

batch_size = 8


class atnet(nn.Module):
    def __init__(self, num_channel=256, num_layer=4):
        super(atnet, self).__init__()
        self.softattention = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Softmax(dim=1),
        )
        # self.conv = nn.Conv2d(256, 256, kernel_size=3,padding=1, stride=1),
        self.softmax_r = nn.Softmax(dim=-1)

    def forward(self, input):
        x = self.softattention(input)
        x = x * input + input
        # x = x*input + input
        # x=self.conv(x)
        return x


class SAF_Module(nn.Module):  # 合成图降维
    def __init__(self, channel):
        super(SAF_Module, self).__init__()
        self.ave = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, int(channel / 4), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(channel / 4), out_channels=int(channel / 4), kernel_size=3, padding=1, stride=1),
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


class AT(nn.Module):  # 原图降维
    def __init__(self, ):
        super(AT, self).__init__()
        self.atnet = atnet()
        self.SAF_Module = SAF_Module(64)

    def forward(self, x):
        x = self.SAF_Module(x)
        x_chn_weight = self.atnet(x)
        return x_chn_weight


class resblock(nn.Module):
    def __init__(self, channel):
        super(resblock, self).__init__()
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1,
                      stride=1))

    def forward(self, x):
        residual = x
        x = self.conv1_0(x)
        x = self.conv1_0(x)
        x = residual + self.conv1_0(x)

        residual = x
        x = self.conv1_0(x)
        x = self.conv1_0(x)
        x = residual + self.conv1_0(x)

        residual = x
        x = self.conv1_0(x)
        x = self.conv1_0(x)
        x = residual + self.conv1_0(x)
        return x


class AT_weight(nn.Module):  # 合成图降维
    def __init__(self, channel):
        super(AT_weight, self).__init__()
        self.resblock = resblock(64)
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=3, padding=1,
                      stride=1))

    def forward(self, x, x_chn_weight):
        w = x + (x * x_chn_weight)
        w2 = self.resblock(w)
        w2 = self.conv1_0(w2)
        return w2


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = ((self.beta_min + self.reparam_offset ** 2) ** 0.5)
        self.gamma_bound = self.reparam_offset

        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs

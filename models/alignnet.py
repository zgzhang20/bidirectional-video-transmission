import torch
import torch.nn as nn
import torch.nn.functional as F
from .basics import *
from .offsetcoder import OffsetEncodeNet, OffsetDecodeNet

from models.dcn.deform_conv import DeformConv as DCNv1

from .fusion import noise_attention_Module
class FeatureEncoder(nn.Module):
    '''
    Feature Encoder
    '''

    def __init__(self, nf=out_channel_M):
        super(FeatureEncoder, self).__init__()
        self.conv1_1 = nn.Conv2d(3, nf, 5, 2, 2)
        self.conv1 = nn.Conv2d(nf, nf, 5, 1, 2)
        self.conv2_1 = nn.Conv2d(nf, nf, 5, 2, 2)
        self.conv2 = nn.Conv2d(nf, nf, 5, 1, 2)
        self.conv3_1 = nn.Conv2d(nf, nf, 5, 2, 2)
        self.conv3 = nn.Conv2d(nf, nf, 5, 1, 2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.feature_extraction = Resblocks(nf)
        self.NA_module=noise_attention_Module(nf)
    def forward(self, x, n_var):
        x =self.conv1_1(x)
        x = self.conv1(x)+x
        x = self.NA_module(x, n_var)

        x = self.conv2_1(x)
        x = self.conv2(x) + x
        x = self.NA_module(x, n_var)

        x = self.conv3_1(x)
        x = self.conv3(x)+ x
        x = self.NA_module(x, n_var)
        return self.feature_extraction(x)
        return x

class FeatureDecoder(nn.Module):
    '''
    Feature Decoder
    '''

    def __init__(self, nf=out_channel_M):
        super(FeatureDecoder, self).__init__()
        self.recon_trunk = Resblocks(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.deconv1 = nn.ConvTranspose2d(nf, 3, 5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.recon_trunk(x)
        x = self.deconv1(x)
        return x


class PCD_Align(nn.Module):
    '''
    Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=out_channel_M, groups=8, compressoffset=True):
        super(PCD_Align, self).__init__()
        self.compressoffset = compressoffset

        self.offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        # self.offset_conv2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.offset_encoder = OffsetEncodeNet()
        self.offset_decoder = OffsetDecodeNet()
        # self.deformable_convolution = DCNv1(nf, nf, 3, stride=1, padding=1, dilation=1)
        self.refine_conv1 = nn.Conv2d(nf * 6, nf, 3, 1, 1)  # concat for diff
        self.refine_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        self.FeatureEncoder = FeatureEncoder()

    def MotionEstimation(self, ref_fea, inp_fea):
        # motion estimation
        input_offset = torch.cat((ref_fea, inp_fea), dim=1)
        input_offset = self.lrelu(self.offset_conv1(input_offset))
        input_offset = self.lrelu(self.offset_conv3(input_offset))
        # motion compression
        en_offset = self.offset_encoder(input_offset)
        return en_offset

    def MotionCompensation(self, F_offset, B_offset, key_feart):
        # de_offset = self.offset_decoder(q_offset)

        # key_feart = self.offset_conv2(self.FeatureEncoder(key))
        # print(F_offset.shape,B_offset.shape,key_feart.shape)
        # motion compensation
        # aligned_feature = self.deformable_convolution([ref_fea, de_offset])

        a1 = torch.cat((F_offset[0:2].view(1, -1, key_feart.shape[2], key_feart.shape[3]),
                        key_feart[0:2].view(1, -1, key_feart.shape[2], key_feart.shape[3]),
                        B_offset[0:2].view(1, -1, key_feart.shape[2], key_feart.shape[3])), dim=1)
        a2 = torch.cat((F_offset[2:4].view(1, -1, key_feart.shape[2], key_feart.shape[3]),
                        key_feart[1:3].view(1, -1, key_feart.shape[2], key_feart.shape[3]),
                        B_offset[2:4].view(1, -1, key_feart.shape[2], key_feart.shape[3])), dim=1)
        a3 = torch.cat((F_offset[4:6].view(1, -1, key_feart.shape[2], key_feart.shape[3]),
                        key_feart[2:4].view(1, -1, key_feart.shape[2], key_feart.shape[3]),
                        B_offset[4:6].view(1, -1, key_feart.shape[2], key_feart.shape[3])), dim=1)

        refine_feature = torch.cat((a1, a2, a3), dim=0)

        refine_feature = self.lrelu(self.refine_conv1(refine_feature))
        refine_feature = self.lrelu(self.refine_conv2(refine_feature))
        # aligned_feature = aligned_feature + refine_feature

        return refine_feature

    def forward(self, ref_fea, inp_fea):
        # motion estimation
        input_offset = torch.cat([ref_fea, inp_fea], dim=1)
        input_offset = self.lrelu(self.offset_conv1(input_offset))
        input_offset = self.lrelu(self.offset_conv3(input_offset))

        # motion compression
        # en_offset = self.offset_encoder(input_offset)
        # q_offset = self.Q(en_offset)
        # de_offset = self.offset_decoder(q_offset)

        # motion compensation
        aligned_feature = self.deformable_convolution([ref_fea, input_offset])

        refine_feature = torch.cat([aligned_feature, ref_fea], dim=1)
        refine_feature = self.lrelu(self.refine_conv1(refine_feature))
        refine_feature = self.lrelu(self.refine_conv2(refine_feature))
        aligned_feature = aligned_feature + refine_feature

        return aligned_feature

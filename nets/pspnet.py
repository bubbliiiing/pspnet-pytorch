import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from nets.mobilenetv2 import mobilenetv2
from nets.resnet import resnet50


class Resnet(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(pretrained)

        #--------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,1024和30,30,2048
        #--------------------------------------------------------------------------------------------#
        if dilate_scale == 8:
            model.layer3.apply(partial(self._nostride_dilate, dilate=2))
            model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu1 = model.relu1
        self.conv2 = model.conv2
        self.bn2 = model.bn2
        self.relu2 = model.relu2
        self.conv3 = model.conv3
        self.bn3 = model.bn3
        self.relu3 = model.relu3
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        return x_aux, x

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        #--------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,320和30,30,96
        #--------------------------------------------------------------------------------------------#
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x_aux = self.features[:14](x)
        x = self.features[14:](x_aux)
        return x_aux, x
 
class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        #-----------------------------------------------------#
        #   分区域进行平均池化
        #   30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
        #-----------------------------------------------------#
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])
        
        # 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone="resnet50", pretrained=True, aux_branch=True):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone=="resnet50":
            self.backbone = Resnet(downsample_factor, pretrained)
            aux_channel = 1024
            out_channel = 2048
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   f4为辅助分支    [30,30,96]
            #   o为主干部分     [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor, pretrained)
            aux_channel = 96
            out_channel = 320
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

        #--------------------------------------------------------------#
        #	PSP模块，分区域进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   30,30,320 -> 30,30,80 -> 30,30,21
        #--------------------------------------------------------------#
        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(out_channel//4, num_classes, kernel_size=1)
        )

        self.aux_branch = aux_branch

        if self.aux_branch:
            #---------------------------------------------------#
            #	利用特征获得预测结果
            #   30, 30, 96 -> 30, 30, 40 -> 30, 30, 21
            #---------------------------------------------------#
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel//8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel//8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel//8, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.master_branch)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x_aux, x = self.backbone(x)
        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()

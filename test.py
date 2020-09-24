import torch
from nets.pspnet import PSPNet
from torchsummary import summary

model = PSPNet(num_classes=21,backbone="mobilenet",downsample_factor=16,aux_branch=False,pretrained=False).train().cuda()

summary(model,(3,473,473))
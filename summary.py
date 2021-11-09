#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from torchsummary import summary

from nets.pspnet import PSPNet

if __name__ == "__main__":
    model = PSPNet(num_classes=21, backbone="mobilenet", downsample_factor=16, aux_branch=False, pretrained=False).train().cuda()
    summary(model,(3, 473, 473))

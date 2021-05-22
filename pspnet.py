import colorsys
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.pspnet import PSPNet as pspnet


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image,nw,nh

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、backbone和num_classes都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的model_path、
#   backbone和num_classes数的修改
#--------------------------------------------#
class PSPNet(object):
    _defaults = {
        "model_path"        : 'model_data/pspnet_mobilenetv2.pth',
        "model_image_size"  : (473, 473, 3),
        "backbone"          : "mobilenet",
        "downsample_factor" : 16,
        "num_classes"       : 21,
        "cuda"              : True,
        #--------------------------------#
        #   blend参数用于控制是否
        #   让识别结果和原图混合
        #--------------------------------#
        "blend"             : True,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   True和False都可以尝试一下，有些时候正效果，有些时候负效果，比较玄学
        #   默认设置为预训练数据集中效果比较好的设置方式。
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
    }

    #---------------------------------------------------#
    #   初始化PSPNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.net = pspnet(num_classes=self.num_classes, downsample_factor=self.downsample_factor, pretrained=False, backbone=self.backbone, aux_branch=False)
        self.net = self.net.eval()
        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict, strict=False)
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]
        else:
            # 画框设置不同的颜色
            hsv_tuples = [(x / self.num_classes, 1., 1.)
                        for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))
            self.colors = (0, 0, 0)

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #---------------------------------------------------------#
        image = image.convert('RGB')

        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        #---------------------------------------------------#
        #   进行不失真的resize，添加灰条，进行图像归一化
        #---------------------------------------------------#
        if self.letterbox_image:
            image, nw, nh = letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        else:
            image = image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        images = [np.array(image)/255]
        images = np.transpose(images,(0,3,1,2))

        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0), dim = -1).cpu().numpy().argmax(axis=-1)
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            if self.letterbox_image:
                pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
            
        #------------------------------------------------#
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        #------------------------------------------------#
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:,:,0] += ((pr[:,: ] == c )*( self.colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( self.colors[c][2] )).astype('uint8')

        #------------------------------------------------#
        #   将新图片转换成Image的形式
        #------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h), Image.NEAREST)

        #------------------------------------------------#
        #   将新图片和原图片混合
        #------------------------------------------------#
        if self.blend:
            image = Image.blend(old_img,image,0.7)
        
        return image

    def get_FPS(self, image, test_interval):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        #---------------------------------------------------#
        #   进行不失真的resize，添加灰条，进行图像归一化
        #---------------------------------------------------#
        if self.letterbox_image:
            image, nw, nh = letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        else:
            image = image.convert('RGB')
            image = image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        images = [np.array(image)/255]
        images = np.transpose(images,(0,3,1,2))
        
        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            if self.letterbox_image:
                pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
            
        image = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h),Image.NEAREST)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                #--------------------------------------#
                #   将灰条部分截取掉
                #--------------------------------------#
                if self.letterbox_image:
                    pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
                
            image = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h),Image.NEAREST)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
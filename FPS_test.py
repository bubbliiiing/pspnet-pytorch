import colorsys
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.autograd import Variable

from pspnet import PSPNet

'''
该FPS测试不包括前处理（归一化与resize部分）、绘图的部分。
使用'img/street.jpg'图片进行测试，该测试方法参考库https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
video.py里面测试的FPS会低于该FPS，因为摄像头的读取频率有限，而且处理过程包含了前处理和绘图部分。
'''

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
    
class FPS_Pspnet(PSPNet):
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
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
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
        
pspnet = FPS_Pspnet()
test_interval = 100
img = Image.open('img/street.jpg')
tact_time = pspnet.get_FPS(img, test_interval)
print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

from pspnet import PSPNet
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F  
import numpy as np
import colorsys
import torch
import copy
import os

class miou_Pspnet(PSPNet):
    def detect_image(self, image):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image, nw, nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        images = [np.array(image)/255]
        images = np.transpose(images,(0,3,1,2))
        
        with torch.no_grad():
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
        
        pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
        
        image = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h),Image.NEAREST)

        return image

pspnet = miou_Pspnet()

image_ids = open(r"VOCdevkit\VOC2007\ImageSets\Segmentation\val.txt",'r').read().splitlines()

if not os.path.exists("./miou_pr_dir"):
    os.makedirs("./miou_pr_dir")

for image_id in image_ids:
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    image = pspnet.detect_image(image)
    image.save("./miou_pr_dir/" + image_id + ".png")
    print(image_id," done!")

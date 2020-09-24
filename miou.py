import numpy as np
import argparse
import json
from PIL import Image
from os.path import join

# 设标签宽W，长H
def fast_hist(a, b, n):
    # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)
    k = (a >= 0) & (a < n)
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    # 返回中，写对角线上的为分类正确的像素点
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  

def per_class_PA(hist):
    # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
    return np.diag(hist) / hist.sum(1)

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):  
    # 计算mIoU的函数
    print('Num classes', num_classes)  
    ## 1
    hist = np.zeros((num_classes, num_classes))
    
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]  # 获得验证集标签路径列表，方便直接读取
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]  # 获得验证集图像分割结果路径列表，方便直接读取

    # 读取每一个（图片-标签）对
    for ind in range(len(gt_imgs)): 
        # 读取一张图像分割结果，转化成numpy数组
        pred = np.array(Image.open(pred_imgs[ind]))  
        # 读取一张对应的标签，转化成numpy数组
        label = np.array(Image.open(gt_imgs[ind]))  

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue
        # 对一张图片计算19×19的hist矩阵，并累加
        hist += fast_hist(label.flatten(), pred.flatten(),num_classes)  
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if ind > 0 and ind % 10 == 0:  
            print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(ind, len(gt_imgs),
                                                    100 * np.mean(per_class_iu(hist)),
                                                    100 * np.mean(per_class_PA(hist))))
    # 计算所有验证集图片的逐类别mIoU值
    mIoUs   = per_class_iu(hist)
    mPA     = per_class_PA(hist)
    # 逐类别输出一下mIoU值
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tmIou-' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA-' + str(round(mPA[ind_class] * 100, 2)))
    # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))  
    return mIoUs


if __name__ == "__main__":
    gt_dir = "./VOCdevkit/VOC2007/SegmentationClass"
    pred_dir = "./miou_pr_dir"
    png_name_list = open(r"VOCdevkit\VOC2007\ImageSets\Segmentation\val.txt",'r').read().splitlines() 
    
    num_classes = 21
    name_classes = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes)  # 执行计算mIoU的函数

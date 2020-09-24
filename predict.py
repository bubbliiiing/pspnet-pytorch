#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from pspnet import PSPNet
from PIL import Image

pspnet = PSPNet()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = pspnet.detect_image(image)
        r_image.show()


"""
读取图片，转灰度，resize到28
传入mnist模型中predict

@Yuan Sheng
"""

from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np
from PIL import Image

# 读取训练好的模型
from mnist_model import *
model = MnistModel('models/mnist2/mnist2.tfl')
# 读取图片转成灰度格式
img = Image.open('1.bmp').convert('L')

# resize的过程
if img.size[0] != 28 or img.size[1] != 28:
    img = img.resize((28, 28))

# 暂存像素值的一维数组
arr = []

for i in range(28):
    for j in range(28):
        # mnist 里的颜色是0代表白色（背景），1.0代表黑色
        pixel = 1.0 - float(img.getpixel((j, i)))/255.0
        # pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
        arr.append(pixel)

arr1 = np.array(arr).reshape((1, 28, 28, 1))
print (model.predict(arr1))
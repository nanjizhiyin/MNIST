#encoding: utf-8
# Copyright 2016 Niek Temme.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#  https://niektemme.com/2016/02/21/tensorflow-handwriting/
# ==============================================================================

"""Predict a handwritten integer (MNIST beginners).

Script requires
1) saved model (model.ckpt file) in the same location as the script is run from.
(requried a model created in the MNIST beginners tutorial)
2) one argument (png file location of a handwritten integer)

Documentation at:
http://niektemme.com/ @@to do
"""

#import modules
from __future__ import print_function
import sys
import tensorflow as tf
from PIL import Image, ImageFilter
import numpy as np

def predictint(imvalue):
    """
    This function returns the predicted integer.
    The input is the pixel values from the imageprepare() function.
    """

    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    saver.restore(sess, "ckpt1/model1.ckpt")
    prediction=tf.argmax(y,1)
    return prediction.eval(feed_dict={x: [imvalue]}, session=sess)


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    # # 读取图片转成灰度格式
    # img = Image.open(argv).convert('L')

    # # resize的过程
    # if img.size[0] != 28 or img.size[1] != 28:
    #     img = img.resize((28, 28))

    # # 暂存像素值的一维数组
    # arr = []

    # for i in range(28):
    #     for j in range(28):
    #         # mnist 里的颜色是0代表白色（背景），1.0代表黑色
    #         pixel = 1.0 - float(img.getpixel((j, i)))/255.0
    #         # pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
    #         # print pixel
    #         print('%3d'%(pixel*100), end='')
    #         arr.append(pixel)
    #     print('')

    # # arr1 = np.array(arr).reshape((1, 28, 28, 1))
    # return arr


    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels

    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
        # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas

    newImage.save("sample.png")

    tv = list(newImage.getdata()) #get pixel values

    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv]
    return tva

def main(argv):
    """
    Main function.
    """
    imvalue = imageprepare(argv)
    predint = predictint(imvalue)
    print ("get num:")
    print (predint)
    print (predint[0]) #first value in list

if __name__ == "__main__":
    main("image/im6.png")
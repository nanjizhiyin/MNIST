#encoding: utf-8
# 源码网址 https://niektemme.com/2016/02/21/tensorflow-handwriting/
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import scipy.ndimage

# 计算分类softmax会将xW+b分成10类，对应0-9
W = tf.Variable(tf.zeros([784, 10]))  #权重
b = tf.Variable(tf.zeros([10]))  #偏置

# 输入
#输入占位符（每张手写数字784个像素点）
x = tf.placeholder("float", [None, 784])
# 输入矩阵x与权重矩阵W相乘，加上偏置矩阵b，然后求softmax（sigmoid函数升级版，可以分成多类）
y = tf.nn.softmax(tf.matmul(x, W) + b)
#输入占位符（这张手写数字具体代表的值，0-9对应矩阵的10个位置）
y_ = tf.placeholder("float", [None, 10])

# 损失
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 使用梯度下降法（步长0.01），来使偏差和最小
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 初始化变量
sess = tf.InteractiveSession()
# 保存器
saver = tf.train.Saver()

# 判断模型保存路径是否存在，不存在就创建
ckptPath = "ckpt1/"
if not os.path.exists(ckptPath):
    os.mkdir(ckptPath)

if os.path.exists(ckptPath + 'checkpoint'):  #判断模型是否存在
    saver.restore(sess, ckptPath + 'model1.ckpt')  #存在就从模型中恢复变量
else:
    tf.global_variables_initializer().run()  #不存在就初始化变量


# 训练自己的图片
tmpData = np.vectorize(lambda x: 255 - x)(
    np.ndarray.flatten(scipy.ndimage.imread("4.png", flatten=True)))

result = sess.run(tf.argmax(y, 1), feed_dict={x: [tmpData]})

print("训练结束" + result)

# 保存训练模型
save_path = saver.save(sess, ckptPath + 'model1.ckpt')
print("模型保存：%s" % (save_path))

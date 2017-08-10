# 源码网址 https://niektemme.com/2016/02/21/tensorflow-handwriting/
from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.ndimage

# 获取数据（如果存在就读取，不存在就下载完再读取）
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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

for i in range(10):  # 训练20000次
    batch_xs, batch_ys = mnist.train.next_batch(10)  # 随机取100个手写数字图片

    # print("图片：%s, 标签:%s" % (batch_xs, batch_ys))

    # # 查看第一张图片
    # batch_x = batch_xs[0]
    # for m in range(28):
    #     for j in range(28):
    #         pixel = batch_x[m*28 + j]
    #         print('%3d' % (pixel*100), end='')
    #     print('')


    _, loss_value = sess.run(
        [train_step,loss],
        feed_dict={x: batch_xs,
                   y_: batch_ys})  # 执行梯度下降算法，输入值x：batch_xs，输入值y：batch_ys
    print("当前训练损失：%s" % (loss_value))

    # # 每50次保存一次模型
    # if (i % 50 == 0):
    #     #保存模型到tmp/model.ckpt，注意一定要有一层文件夹，否则保存不成功！！！
    #     save_path = saver.save(
    #         sess, ckptPath +
    #         'model1.ckpt')
    #     print("模型保存：%s 当前训练损失：%s" % (save_path, loss_value))

print("训练MNIST图片结束")

# 训练自己的图片
tmpData = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("4.png", flatten=True)))

result = sess.run(tf.argmax(y, 1), feed_dict={x: [tmpData]})

print("训练结束"+result)

# 保存训练模型
save_path = saver.save(sess, ckptPath + 'model1.ckpt')
print("模型保存：%s" % (save_path))

# # 计算训练精度
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print(sess.run(
#     accuracy, feed_dict={x: mnist.test.images,
#                          y_: mnist.test.labels}))  #运行精度图，x和y_从测试手写图片中取值

# 关闭session
# sess.close()
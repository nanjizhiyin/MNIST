import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

# 获取数据（如果存在就读取，不存在就下载完再读取）
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X, Y, testX, testY = mnist.train.next_batch(1)  # 随机取100个手写数字图片
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
# 取出测试集中第一张图片的像素，reshape成28*28
test_x_0 = testX[0].reshape([28, 28])

# 新建一张图片，把test_x_0的像素写入
img = Image.new('L', (28, 28), 255)
for i in range(28):
    for j in range(28):
        img.putpixel((j, i), 255 - int(test_x_0[i][j] * 255.0))

# 保存图片以查看，也可以直接img.show()
img.save('test_save.png')

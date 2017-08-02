import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取mnist张量集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 我们通过操作符号变量来描述这些可交互的操作单元，可以用下面的方式创建一个
# x不是一个特定的值，而是一个占位符placeholder
x = tf.placeholder("float", [None, 784])

# 模型也需要权重值和偏置量
# 一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 训练模型

# 为了实现交叉熵，我们需要先添加一个新的占位符来输入正确答案
y_ = tf.placeholder(tf.float32, [None, 10])

# 然后我们可以实现交叉熵函数：
# tf.reduce_mean计算批次中所有示例的平均值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.05的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# 在一个Session里面启动我们的模型，并且初始化变量
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 我们来训练 - 我们将运行1000次训练步骤！
for _ in range(10):
    # 循环的每一步，我们从训练集中得到一百个随机数据点的“批次”。我们运行train_step饲料中的批次数据来代替placeholders
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#评估我们的模型

# 我们的模型做得如何？
# 那么首先我们来弄清楚我们预测正确的标签。tf.argmax 是一个非常有用的功能，
# 它给出沿某个轴的张量中最高条目的索引。
# 例如，tf.argmax(y,1)我们的模型认为是每个输入最有可能的标签，
# tf.argmax(y_,1)而是正确的标签。
# 我们可以tf.equal用来检查我们的预测是否符合真相。
ymax = tf.argmax(y, 1)
_ymax = tf.argmax(y_, 1)
print(ymax)
print(_ymax)
correct_prediction = tf.equal(ymax, _ymax)

# 这给了我们一个布尔的列表。为了确定哪个部分是正确的，
# 我们转换为浮点数，然后取平均值。
# 例如， [True, False, True, True]会变成[1,0,1,1]哪一个0.75。

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

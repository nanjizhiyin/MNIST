#encoding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def input_pipeline(file_name, label):

    input_name = "file_reader"
    file_reader = tf.read_file(file_name, input_name)

    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels = 3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                            name='jpeg_reader')

    new_shape = [28, 28]

    float_caster = tf.cast(image_reader, tf.float32)

    # 图片
    image_batch = tf.image.resize_images(float_caster, new_shape)

    # # 图片
    # dims_expander = tf.expand_dims(float_caster, 0)
    # image_batch = tf.image.resize_bilinear(dims_expander, new_shape)

    batch_size = 1
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    # 通过随机洗牌来形成批次。
    image_batch, label_batch = tf.train.shuffle_batch(
        [image_batch, label],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch

# 开始

root_dir = "/Users/xuexin/Documents/GitHub/MNIST/TF/"

# 图片
filenames1 = root_dir+"4.jpeg"
# 标签
label = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0,]

image_batch, label_batch = input_pipeline(filenames1, label)
print("image_batch=")
print(image_batch)
print("label_batch=")
print(label_batch)


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    images = image_batch.eval()
    print(images.shape)
    labels = label_batch.eval()
    print(labels.shape)


    coord.request_stop()
    coord.join(threads)
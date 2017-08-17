#encoding: utf-8
import tensorflow as tf

def read_my_file_format(filename_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    example = tf.image.decode_jpeg(value, channels=1)
    # example, label = tf.some_decoder(value)
    casted_example = tf.cast(example, tf.float32)
    new_shape = [28, 28];
    resized_example = tf.image.resize_images(casted_example, new_shape)
    return resized_example

def input_pipeline(filenames, batch_size):
    filename_queue = tf.train.string_input_producer(filenames)
    # 图片数据
    single_image = read_my_file_format(filename_queue)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    # 通过随机洗牌来形成批次。
    image_batch, label_batch = tf.train.shuffle_batch(
        [single_image, '4'],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch

root_dir = "/Users/xuexin/Documents/GitHub/MNIST/TF/"
filenames1 = [root_dir+"4.jpeg",root_dir+"12.jpg"]
# filenames2 = [(root_dir+"9.bmp" % i) for i in range(10)]

batch_size = 2
image_batch, label_batch = input_pipeline(filenames1, batch_size)


init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    images = image_batch.eval()
    labels = label_batch.eval()
    print(images.shape)
    print(labels.shape)
    print(labels)

    coord.request_stop()
    coord.join(threads)
import tensorflow as tf
# Create two variables.
weights = tf.Variable(
    tf.random_normal([784, 200], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")

print(weights)
print(biases)
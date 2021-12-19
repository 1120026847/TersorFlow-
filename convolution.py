# p212
import tensorflow as tf

xs=tf.random.truncated_normal(shape=[50,32,32,32])
out =tf.keras.layers.Conv2D(64,3,padding="SAME")(xs)
print(out.shape)

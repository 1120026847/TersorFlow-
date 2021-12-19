# p 216
import tensorflow as tf
xs=tf.random.truncated_normal(shape=[50,32,32,32])
out_1 =tf.keras.layers.Dense(32)(xs)
print(out_1)
print(out_1.shape)

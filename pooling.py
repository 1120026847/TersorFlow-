# p217
import tensorflow as tf
xs=tf.random.truncated_normal(shape=[50,32,32,32])
out=tf.keras.layers.AveragePooling2D(strides=[1,1])(xs)
print(out.shape)
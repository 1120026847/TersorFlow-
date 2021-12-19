import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
keras = tf.keras
import numpy as np
path = "./cifar-100-python"
from tensorflow.python.keras.datasets.cifar import load_batch
fpath = os.path.join(path,'train')
x_train, y_train =load_batch(fpath, label_key='fine'+'_labels')
x_train=x_train[1000:2000]
y_train=y_train[1000:2000]
fpath = os.path.join(path,'test')
x_test, y_test =load_batch(fpath, label_key='fine'+'_labels')
x_test=x_test[1000:2000]
y_test=y_test[1000:2000]

x_train = tf.transpose(x_train,[0,2,3,1])
y_train = np.float32(tf.keras.utils.to_categorical(y_train,num_classes=100))
x_test = tf.transpose(x_test,[0,2,3,1])
y_test = np.float32(tf.keras.utils.to_categorical(y_test,num_classes=100))

batch_size =24
train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(batch_size*10).batch(batch_size)

import resnet_model
model = resnet_model.resnet_Model()
model.compile(optimizer=tf.optimizers.Adam(1e-2),loss=tf.losses.categorical_crossentropy,metrics = ['accuracy'])
model.fit(train_data,epochs=5)
score = model.evaluate(x_test,y_test)
print("last score:",score)
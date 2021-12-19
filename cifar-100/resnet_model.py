import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
keras = tf.keras
def identity_block(input_tensor, out_dim):
    conv_1 = keras.layers.Conv2D(out_dim // 4, 1,padding="SAME",activation=tf.nn.relu)(input_tensor)
    conv_2 = keras.layers.BatchNormalization()(conv_1)
    conv_3 = keras.layers.Conv2D(out_dim // 4, 3, padding="SAME", activation=tf.nn.relu)(conv_2)
    conv_4 = keras.layers.BatchNormalization()(conv_3)
    conv_5 = keras.layers.Conv2D(out_dim,1,padding="SAME")(conv_4)
    out = keras.layers.Add()([input_tensor,conv_5])
    out = tf.nn.relu(out)
    return out
def resnet_Model(n_dim=10):
    input_xs = keras.layers.Input([32,32,3])
    conv_1 = keras.layers.Conv2D(64,3,padding="SAME",activation=tf.nn.relu) (input_xs)


    identity_1 = tf.keras.layers.Conv2D(64,3,padding="SAME",activation=tf.nn.relu) (conv_1)
    identity_1 = tf.keras.layers.BatchNormalization()(identity_1)
    for _ in range(3):
        identity_1 = identity_block(identity_1,64)


    identity_2 = tf.keras.layers.Conv2D(128,3,padding="SAME",activation=tf.nn.relu) (identity_1)
    identity_2 = tf.keras.layers.BatchNormalization()(identity_2)
    for _ in range(4):
        identity_2 = identity_block(identity_2,128)


    identity_3 = tf.keras.layers.Conv2D(256,3,padding="SAME",activation=tf.nn.relu) (identity_2)
    identity_3 = tf.keras.layers.BatchNormalization()(identity_3)
    for _ in range(6):
        identity_3 = identity_block(identity_3,256)


    identity_4 = tf.keras.layers.Conv2D(512,3,padding="SAME",activation=tf.nn.relu) (identity_3)
    identity_4 = tf.keras.layers.BatchNormalization()(identity_4)
    for _ in range(3):
        identity_4 = identity_block(identity_4,512)

    flat = keras.layers.Flatten()(identity_4)
    flat = keras.layers.Dropout(0.217)(flat)
    dense = keras.layers.Dense(1024,activation=tf.nn.relu)(flat)
    dense =keras.layers.BatchNormalization()(dense)
    logits = keras.layers.Dense(100,activation=tf.nn.softmax)(dense)

    model=keras.Model(inputs=input_xs,outputs=logits)
    return model

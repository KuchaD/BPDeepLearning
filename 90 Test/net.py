import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import math

from tensorflow.core import input_data

network = input_data(shape=[None,28,28, 3])
conv1_7_7 = tf.layers.conv2d(network, 64, 7, strides=2, activation=tf.nn.relu, name = 'conv1_7_7_s2')
pool1_3_3 = tf.layers.max_pooling2d(conv1_7_7, 3,strides=2)
pool1_3_3 = tf.layers.batch_normalization(pool1_3_3)
conv2_3_3_reduce = tf.layers.conv2d(pool1_3_3, 64,1, activation=tf.nn.relu,name = 'conv2_3_3_reduce')
conv2_3_3 = tf.layers.conv2d(conv2_3_3_reduce, 192,3, activation=tf.nn.relu, name='conv2_3_3')
conv2_3_3 = tf.layers.batch_normalization(conv2_3_3)
pool2_3_3 = tf.layers.max_pooling2d(conv2_3_3,pool_size=3, strides=2, name='pool2_3_3_s2')


inception_3a_1_1 = tf.layers.conv2d(pool2_3_3, 64, 1, activation=tf.nn.relu, name='inception_3a_1_1')
inception_3a_3_3_reduce = tf.layers.conv2d(pool2_3_3, 96,1, activation=tf.nn.relu, name='inception_3a_3_3_reduce')
inception_3a_3_3 = tf.layers.conv2d(inception_3a_3_3_reduce,filters = 128,kernel_size =3,activation=tf.nn.relu, name = 'inception_3a_3_3')
inception_3a_5_5_reduce = tf.layers.conv2d(pool2_3_3,filters = 16, kernel_size=1,activation=tf.nn.relu, name ='inception_3a_5_5_reduce' )
inception_3a_5_5 = tf.layers.conv2d(inception_3a_5_5_reduce, filters = 32, kernel_size = 5, activation=tf.nn.relu, name= 'inception_3a_5_5')
inception_3a_pool = tf.layers.max_pooling2d(pool2_3_3, pool_size=3, strides=1, )
inception_3a_pool_1_1 = tf.layers.conv2d(inception_3a_pool, 32, 1, activation=tf.nn.relu, name='inception_3a_pool_1_1')

# merge the inception_3a__
inception_3a_output = tf.concat([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],3, name='inception_3a_output')

inception_3b_1_1 = tf.layers.conv2d(inception_3a_output, 128,1,activation=tf.nn.relu, name= 'inception_3b_1_1' )
inception_3b_3_3_reduce = tf.layers.conv2d(inception_3a_output, 128, 1, activation=tf.nn.relu, name='inception_3b_3_3_reduce')
inception_3b_3_3 = tf.layers.conv2d(inception_3b_3_3_reduce, 192, 3,  activation=tf.nn.relu,name='inception_3b_3_3')
inception_3b_5_5_reduce = tf.layers.conv2d(inception_3a_output, 32, 1, activation=tf.nn.relu, name = 'inception_3b_5_5_reduce')
inception_3b_5_5 = tf.layers.conv2d(inception_3b_5_5_reduce, 96, 5,  name = 'inception_3b_5_5')
inception_3b_pool = tf.layers.max_pooling2d(inception_3a_output, pool_size=3, strides=1,  name='inception_3b_pool')
inception_3b_pool_1_1 = tf.layers.conv2d(inception_3b_pool, 64, 1,activation=tf.nn.relu, name='inception_3b_pool_1_1')

#merge the inception_3b_*
inception_3b_output = tf.concat([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],3, name='inception_3b_output')

pool3_3_3 = tf.layers.max_pooling2d(inception_3b_output, pool_size=3, strides=2, name='pool3_3_3')
inception_4a_1_1 = tf.layers.conv2d(pool3_3_3, 192, 1, activation=tf.nn.relu, name='inception_4a_1_1')
inception_4a_3_3_reduce = tf.layers.conv2d(pool3_3_3, 96, 1, activation=tf.nn.relu, name='inception_4a_3_3_reduce')
inception_4a_3_3 = tf.layers.conv2d(inception_4a_3_3_reduce, 208, 3,  activation=tf.nn.relu, name='inception_4a_3_3')
inception_4a_5_5_reduce = tf.layers.conv2d(pool3_3_3, 16, 1, activation=tf.nn.relu, name='inception_4a_5_5_reduce')
inception_4a_5_5 = tf.layers.conv2d(inception_4a_5_5_reduce, 48, 5,  activation=tf.nn.relu, name='inception_4a_5_5')
inception_4a_pool = tf.layers.max_pooling2d(pool3_3_3, pool_size=3, strides=1,  name='inception_4a_pool')
inception_4a_pool_1_1 = tf.layers.conv2d(inception_4a_pool, 64, 1, activation=tf.nn.relu, name='inception_4a_pool_1_1')


inception_4a_output = tf.concat([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],3, name='inception_4a_output')


inception_4b_1_1 = tf.layers.conv2d(inception_4a_output, 160, 1, activation=tf.nn.relu, name='inception_4a_1_1')
inception_4b_3_3_reduce = tf.layers.conv2d(inception_4a_output, 112, 1, activation=tf.nn.relu, name='inception_4b_3_3_reduce')
inception_4b_3_3 = tf.layers.conv2d(inception_4b_3_3_reduce, 224, 3, activation=tf.nn.relu, name='inception_4b_3_3')
inception_4b_5_5_reduce = tf.layers.conv2d(inception_4a_output, 24, 1, activation=tf.nn.relu, name='inception_4b_5_5_reduce')
inception_4b_5_5 = tf.layers.conv2d(inception_4b_5_5_reduce, 64, 5,  activation=tf.nn.relu, name='inception_4b_5_5')
inception_4b_pool = tf.layers.max_pooling2d(inception_4a_output, pool_size=3, strides=1,  name='inception_4b_pool')
inception_4b_pool_1_1 = tf.layers.conv2d(inception_4b_pool, 64, 1, activation=tf.nn.relu, name='inception_4b_pool_1_1')

                
inception_4b_output = tf.concat([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],3, name='inception_4b_output')


inception_4c_1_1 = tf.layers.conv2d(inception_4b_output, 128, 1, activation=tf.nn.relu,name='inception_4c_1_1')
inception_4c_3_3_reduce = tf.layers.conv2d(inception_4b_output, 128, 1, activation=tf.nn.relu, name='inception_4c_3_3_reduce')
inception_4c_3_3 = tf.layers.conv2d(inception_4c_3_3_reduce, 256,  3, activation=tf.nn.relu, name='inception_4c_3_3')
inception_4c_5_5_reduce = tf.layers.conv2d(inception_4b_output, 24, 1, activation=tf.nn.relu, name='inception_4c_5_5_reduce')
inception_4c_5_5 = tf.layers.conv2d(inception_4c_5_5_reduce, 64,  5, activation=tf.nn.relu, name='inception_4c_5_5')
inception_4c_pool = tf.layers.max_pooling2d(inception_4b_output, pool_size=3, strides=1)
inception_4c_pool_1_1 = tf.layers.conv2d(inception_4c_pool, 64, 1, activation=tf.nn.relu, name='inception_4c_pool_1_1')

inception_4c_output = tf.concat([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],3, name='inception_4c_output')

inception_4d_1_1 = tf.layers.conv2d(inception_4c_output, 112, 1, activation=tf.nn.relu, name='inception_4d_1_1')
inception_4d_3_3_reduce = tf.layers.conv2d(inception_4c_output, 144, 1, activation=tf.nn.relu, name='inception_4d_3_3_reduce')
inception_4d_3_3 = tf.layers.conv2d(inception_4d_3_3_reduce, 288, 3, activation=tf.nn.relu, name='inception_4d_3_3')
inception_4d_5_5_reduce = tf.layers.conv2d(inception_4c_output, 32, 1, activation=tf.nn.relu, name='inception_4d_5_5_reduce')
inception_4d_5_5 = tf.layers.conv2d(inception_4d_5_5_reduce, 64, 5,  activation=tf.nn.relu, name='inception_4d_5_5')
inception_4d_pool = tf.layers.max_pooling2d(inception_4c_output, pool_size=3, strides=1,  name='inception_4d_pool')
inception_4d_pool_1_1 = tf.layers.conv2d(inception_4d_pool, 64, 1, activation=tf.nn.relu, name='inception_4d_pool_1_1')


        
inception_4d_output = tf.concat([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],3, name='inception_4d_output')

inception_4e_1_1 = tf.layers.conv2d(inception_4d_output, 256, 1, activation=tf.nn.relu, name='inception_4e_1_1')
inception_4e_3_3_reduce = tf.layers.conv2d(inception_4d_output, 160, 1, activation=tf.nn.relu, name='inception_4e_3_3_reduce')
inception_4e_3_3 = tf.layers.conv2d(inception_4e_3_3_reduce, 320, 3, activation=tf.nn.relu, name='inception_4e_3_3')
inception_4e_5_5_reduce = tf.layers.conv2d(inception_4d_output, 32, 1, activation=tf.nn.relu, name='inception_4e_5_5_reduce')
inception_4e_5_5 = tf.layers.conv2d(inception_4e_5_5_reduce, 128,  5, activation=tf.nn.relu, name='inception_4e_5_5')
inception_4e_pool = tf.layers.max_pooling2d(inception_4d_output, pool_size=3, strides=1,  name='inception_4e_pool')
inception_4e_pool_1_1 = tf.layers.conv2d(inception_4e_pool, 128, 1, activation=tf.nn.relu, name='inception_4e_pool_1_1')


inception_4e_output = tf.concat([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],3, name='inception_4e_output')

pool4_3_3 = tf.layers.max_pooling2d(inception_4e_output, pool_size=3, strides=2, name='pool_3_3')


inception_5a_1_1 = tf.layers.conv2d(pool4_3_3, 256, 1, activation=tf.nn.relu, name='inception_5a_1_1')
inception_5a_3_3_reduce = tf.layers.conv2d(pool4_3_3, 160, 1, activation=tf.nn.relu, name='inception_5a_3_3_reduce')
inception_5a_3_3 = tf.layers.conv2d(inception_5a_3_3_reduce, 320, 3, activation=tf.nn.relu, name='inception_5a_3_3')
inception_5a_5_5_reduce = tf.layers.conv2d(pool4_3_3, 32, 1, activation=tf.nn.relu, name='inception_5a_5_5_reduce')
inception_5a_5_5 = tf.layers.conv2d(inception_5a_5_5_reduce, 128, 5,  activation=tf.nn.relu, name='inception_5a_5_5')
inception_5a_pool = tf.layers.max_pooling2d(pool4_3_3, pool_size=3, strides=1,  name='inception_5a_pool')
inception_5a_pool_1_1 = tf.layers.conv2d(inception_5a_pool, 128, 1,activation=tf.nn.relu, name='inception_5a_pool_1_1')
        
inception_5a_output = tf.concat([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1],3, name='inception_5a_output')

inception_5b_1_1 = tf.layers.conv2d(inception_5a_output, 384, 1,activation=tf.nn.relu, name='inception_5b_1_1')
inception_5b_3_3_reduce = tf.layers.conv2d(inception_5a_output, 192, 1, activation=tf.nn.relu, name='inception_5b_3_3_reduce')
inception_5b_3_3 = tf.layers.conv2d(inception_5b_3_3_reduce, 384,  3,activation=tf.nn.relu, name='inception_5b_3_3')
inception_5b_5_5_reduce = tf.layers.conv2d(inception_5a_output, 48, 1, activation=tf.nn.relu, name='inception_5b_5_5_reduce')
inception_5b_5_5 = tf.layers.conv2d(inception_5b_5_5_reduce,128, 5,  activation=tf.nn.relu, name='inception_5b_5_5' )
inception_5b_pool = tf.layers.max_pooling2d(inception_5a_output, pool_size=3, strides=1,  name='inception_5b_pool')
inception_5b_pool_1_1 = tf.layers.conv2d(inception_5b_pool, 128, 1, activation=tf.nn.relu, name='inception_5b_pool_1_1')

        
inception_5b_output = tf.concat([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1],3, name='inception_5b_output')

pool5_7_7 = tf.layers.average_pooling2d(inception_5b_output, pool_size=7, strides=1)
pool5_7_7 = tf.layers.dropout(pool5_7_7, 0.4)
logits = pool5_7_7
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers import LSTM, Dense, Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
import sys
from numpy import genfromtxt
import pandas as pd
#import tensorflow.compat.v1 as tf
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
tf.compat.v1.enable_eager_execution()


#matplotlib inline
#load_ext autoreload
#autoreload 2


def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist- neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###

    return loss



np.set_printoptions(threshold=sys.maxsize)

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:", FRmodel.count_params())



# GRADED FUNCTION: triplet_loss



### testing triplet_loss function


with tf.compat.v1.Session() as test:
    tf.compat.v1.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.compat.v1.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.compat.v1.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.compat.v1.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.eval()))

####loading the facenet model ;)

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)





#### my_database
#database = {}
#database["fadwa"] = img_to_encoding("images/fadwa.jpg", FRmodel)
#database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
#database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
#database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
#database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
#database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
#database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
#database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
#database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
#database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
#database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
#database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)




#verify("images/camera_0.jpg", "fadwa", database, FRmodel)

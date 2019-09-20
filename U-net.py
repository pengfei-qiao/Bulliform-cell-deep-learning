# provide learning rate and saved model name, also disabled setting random seeds for averaging models later
# convert images black and white
# conver images to boolean format

import os, sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage import img_as_bool

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.initializers import he_normal
from keras.utils import plot_model
from keras import backend as K

import tensorflow as tf

# from numpy.random import seed
# seed(12345)
# from tensorflow import set_random_seed
# set_random_seed(12345)

# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

img_w = 480 # image width
img_h = 480 # image height
img_c = 1 # image channels
train_path = './train/'
test_path = './test/'

train_ids = sorted(next(os.walk(train_path))[2])
test_ids = sorted(next(os.walk(test_path))[2])

X_train = np.array([1 - img_as_bool(imread(train_path + i, as_gray = True)[:480,:480].reshape(img_h, img_w, img_c)) for i in train_ids if '.png' in i and 'Y-' not in i])
Y_train = np.array([1 - img_as_bool(imread(train_path + i, as_gray = True)[:480,:480].reshape(img_h, img_w, img_c)).astype(np.bool) for i in train_ids if 'Y-' in i])
X_test = np.array([1 - img_as_bool(imread(test_path + i, as_gray = True)[:480,:480].reshape(img_h, img_w, img_c)) for i in test_ids if '.png' in i and 'Y-' not in i])
Y_test = np.array([1 - img_as_bool(imread(test_path + i, as_gray = True)[:480,:480].reshape(img_h, img_w, img_c)).astype(np.bool) for i in test_ids if 'Y-' in i])

# center training data
X_train_mean = np.mean([np.ravel(i) for i in X_train],axis=0).reshape(img_h, img_w, img_c) 
X_train = np.array([i - X_train_mean for i in X_train])

# define metric - Dice similary coeffcient
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# compile model
inputs = Input((img_h, img_w, img_c))

conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv5)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv6)

up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv7)

up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv8)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer = 'he_normal', padding='same')(conv9)

conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

model = Model(inputs=[inputs], outputs=[conv10])

model.compile(optimizer=Adam(lr=float(sys.argv[1])), loss='binary_crossentropy', metrics=[dice_coef])

# plot_model(model, to_file='model.png')

# model.summary()

# # check size of tensor
# K.int_shape(conv5) # None in the output shape seems like to be the placeholder's dimension
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-lr-%s-%s-white.h5' %(str(sys.argv[1]),str(sys.argv[2])), verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=2000, 
                    callbacks=[earlystopper, checkpointer])


# # load saved model
# model = load_model('model-lr-1e-2-1.h5', custom_objects = {'dice_coef':dice_coef})

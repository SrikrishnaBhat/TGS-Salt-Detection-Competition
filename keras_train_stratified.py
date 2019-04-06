
# coding: utf-8

# In[1]:


import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2

from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import metrics

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split

from tgs_nets.u_net import build_u_net_one_less, build_u_net_dropouts, build_u_net_vgg


# Set some parameters
im_width = 128
im_height = 128
im_chan = 1
path_train = 'train'
path_test = '.'


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


batch_size = 32
epochs = 70


model_loss = {'mask_output': 'binary_crossentropy', 'depth_output': 'mean_squared_error'}
model_metrics = {'mask_output': mean_iou}


# Build U-Net model
inputs = Input((im_height, im_width, im_chan))
s = Lambda(lambda x: x / 255)(inputs)
outputs = build_u_net_vgg(s)

model = Model(inputs=[inputs], outputs=outputs)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.compile(optimizer='adam', loss=model_loss, metrics=model_metrics)
model.summary()

import os

base_dir = 'trained_weights'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

src_dir = 'stratified_vals'
file_list = os.listdir(src_dir)

file_list.sort()

for i, file_name in enumerate(file_list):
    print("Epoch: {}, Training for: {}".format(i, file_name))
    model_file_name = os.path.join(
        base_dir,
        'model-tgs-salt-double-output-' + file_name + '-{epoch:02d}-{val_loss:.2f}-{val_mask_output_mean_iou:.2f}.h5'
    )
    data_dict = np.load(os.path.join(src_dir, file_name)).item()
    X_train = data_dict['images']
    Y_train = data_dict['masks']

    min_depth = 0
    max_depth = 1000

    depths = np.array(data_dict['depths'])
    if np.max(depths) > 1:
        depths = (depths - min_depth) / (max_depth - min_depth)

    print(X_train.shape, Y_train.shape, depths.shape)

    earlystopper = EarlyStopping(patience=10, verbose=1)
    checkpointer = ModelCheckpoint(model_file_name, monitor='val_loss', mode='min', verbose=1,
                                   save_best_only=True)
    results = model.fit(X_train, [Y_train, depths], validation_split=0.2, batch_size=16, epochs=70,
                        callbacks=[checkpointer])
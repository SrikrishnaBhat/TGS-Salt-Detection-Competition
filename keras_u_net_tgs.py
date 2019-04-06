
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

# In[2]:


# Set some parameters
im_width = 128
im_height = 128
im_chan = 1
path_train = 'train'
path_test = '.'

# In[6]:

# Get and resize train images and masks
# X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
# Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
# print('Getting and resizing train images and masks ... ')
# sys.stdout.flush()
# for n, id_ in enumerate(train_ids):
#     path = path_train
#     img = load_img(os.path.join(path, 'images', id_))
#     x = img_to_array(img)[:,:,0]
#     x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
#     X_train[n] = x
#     mask = img_to_array(load_img(os.path.join(path, 'masks', id_)))[:,:,1]
#     Y_train[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)
#
# print('Done!')

data_dict = np.load('resized_depth_based_train_data.npy').item()
X_train = data_dict['images']
Y_train = data_dict['masks']

min_depth = 0
max_depth = 1000

depths = np.array(data_dict['depths'])
depths = (depths - min_depth)/(max_depth - min_depth)
rows = depths.shape
cols = np.max(depths)



# In[8]:


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

def mape(y_true, y_pred):
    return K.mean(np.abs(y_true - y_pred)/y_true, axis=0)

batch_size = 32
epochs = 70

# In[9]:

model_loss = {'mask_output': 'binary_crossentropy', 'depth_output': 'mean_squared_error'}
model_metrics = {'mask_output': mean_iou}

# Build U-Net model
inputs = Input((im_height, im_width, im_chan))
s = Lambda(lambda x: x / 255) (inputs)

# outputs = build_u_net_dropouts(s)
#
# model = Model(inputs=[inputs], outputs=[outputs])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
# model.summary()
#
#
# # In[ ]:
# earlystopper = EarlyStopping(patience=5, verbose=1)
# checkpointer = ModelCheckpoint('model-tgs-salt-dropout-2.h5', verbose=1, save_best_only=True)
# results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30,
#                     callbacks=[checkpointer])

outputs = build_u_net_vgg(s)

model = Model(inputs=[inputs], outputs=outputs)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.compile(optimizer='adam', loss=model_loss, metrics=model_metrics)
model.summary()

import os

base_dir = 'trained_weights'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

model_file_name = os.path.join(
    base_dir,
    'model-tgs-salt-double-output-{epoch:02d}-{val_loss:.2f}-{val_mask_output_mean_iou:.2f}.h5'
)

earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint(model_file_name, monitor='val_loss', mode='max', verbose=1,
                               save_best_only=True)
results = model.fit(X_train, [Y_train, depths], validation_split=0.2, batch_size=16, epochs=70,
                    callbacks=[checkpointer])

# validation_split = 0.2
# limit = int(validation_split * len(X_train))
# train_X, test_X = X_train[:-limit], X_train[-limit:]
# train_Y, test_Y = Y_train[:-limit], Y_train[-limit:]
# train_depths, test_depths = depths[:-limit], depths[-limit:]
#
# max_metric = 0
#
# print('Total loss, mask_loss, depth_loss, iou, epoch')
#
# for epoch in range(epochs):
#     for i in range(0, len(X_train), batch_size):
#         batch_X = train_X[i:i+batch_size]
#         batch_Y = train_Y[i:i+batch_size]
#         batch_depths = train_depths[i:i+batch_size]
#
#         train_output = model.train_on_batch(batch_X, {'mask_output': batch_Y, 'depth_output': batch_depths})
#         print(train_output)
#     train_output += epoch
#     print(train_output)
#
#     indices = np.arange(0, len(test_X))
#     np.random.shuffle(indices)
#
#     vali_X = test_X[indices[:batch_size]]
#     vali_Y = test_Y[indices[:batch_size]]
#     test_depths = test_depths[indices[:batch_size]]
#
#     pred_Y, _ = model.predict(vali_X)
#     iou_accuracy = mean_iou(vali_Y, pred_Y)
#
#     if max_metric<iou_accuracy:
#         max_metric = iou_accuracy
#         model.save(model_file_name.format(epoch))
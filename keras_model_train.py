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

import tensorflow as tf
import preprocessing.kernels as kerns
from tgs_nets.u_net import build_u_net_normal

data_dict = np.load('upsampled_training_array.npy').item()

im_width = 128
im_height = 128
im_chan = 1

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

X_train = data_dict['train']['images']
Y_train = data_dict['train']['masks']

num_kernels = 6

gabor_kernels = kerns.gabor_kernel(num_kernels)

np.save('gabor_kernels_6.npy', np.asarray(gabor_kernels))

train_images = []#X_train.copy()

for i, image in enumerate(X_train):
    train_images.append(np.asarray(kerns.process_gabor_stacked(image, gabor_kernels)).reshape(im_height, im_width, num_kernels))
    for j in range(num_kernels):
        cv2.imshow('im{}'.format(j), train_images[0][:, :, j])
    k = cv2.waitKey(-1)
    sys.exit(0)

train_images = np.asarray(train_images)
print(train_images.shape)

inputs = Input((im_height, im_width, num_kernels))
s = Lambda(lambda x: x / 255) (inputs)

outputs = build_u_net_normal(s)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()


# In[ ]:
# earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-tgs-salt-gabor-3.h5', verbose=1, save_best_only=True)
results = model.fit(train_images, Y_train, validation_split=0.1, batch_size=8, epochs=90,
                    callbacks=[checkpointer])


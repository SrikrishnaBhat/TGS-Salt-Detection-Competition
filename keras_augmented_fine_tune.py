from keras.models import load_model
import numpy as np
from keras import backend as K

from keras.callbacks import ModelCheckpoint

import tensorflow as tf


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


image_file_list = [
    'flipped_image_normal_lr.npy',
    'flipped_image_normal_ud.npy',
    'flipped_image_normal_full.npy'
]

mask_file_list = [
    'flipped_mask_normal_lr.npy',
    'flipped_mask_normal_ud.npy',
    'flipped_mask_normal_full.npy'
]

image_rows = 128
image_cols = 128
image_chans = 1

image_array = []
for image_file in image_file_list:
    image_array.append(np.load(image_file))
image_array = np.asarray(image_array).reshape(-1, image_rows, image_cols, image_chans)

mask_array = []
for mask_file in mask_file_list:
    mask_array.append(np.load(mask_file))

mask_array = np.asarray(mask_array).reshape(-1, image_rows, image_cols, image_chans)

indices = np.arange(0, image_array.shape[0])
np.random.shuffle(indices)

image_array = image_array[indices, :, :, :]
mask_array = mask_array[indices, :, :, :]

del indices

unet_model = load_model('model-tgs-salt-dropout-2.h5', custom_objects={'mean_iou': mean_iou})

# In[ ]:
# earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-tgs-salt-dropout-2.h5', verbose=1, save_best_only=True)
results = unet_model.fit(image_array, mask_array, validation_split=0.3, batch_size=64, epochs=70,
                    callbacks=[checkpointer])

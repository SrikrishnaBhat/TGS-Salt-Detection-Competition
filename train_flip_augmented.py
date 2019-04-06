from keras.models import load_model
import numpy as np
from keras import backend as K
import sys
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint

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

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python train_flip_augmented.py <augmented_array_index_no>')
        sys.exit(1)

    index = int(sys.argv[1])

    image_file = 'augmented_images_{}.npy'.format(index)
    mask_file = 'augmented_masks_{}.npy'.format(index)

    print('Index: {}, Image File: {}, Mask File: {}'.format(index, image_file, mask_file))
    
    image_array = np.load(os.path.join('train/generated/flipped_images', image_file))
    mask_array = np.load(os.path.join('train/generated/flipped_masks', mask_file))

    image_shape = image_array.shape

    indices = np.arange(0, image_shape[0])
    np.random.shuffle(indices)

    image_array = image_array[indices]
    mask_array = mask_array[indices]

    unet_model = load_model('model-tgs-salt-dropout-2.h5', custom_objects={'mean_iou': mean_iou})

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-tgs-flip-augmented-2.h5', verbose=1, save_best_only=True)
    results = unet_model.fit(image_array, mask_array, validation_split=0.3, batch_size=64, epochs=70,
                        callbacks=[checkpointer])


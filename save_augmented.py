import numpy as np
import os
import cv2
import sys

num_rows = 128
num_cols = 128
num_channels = 1

def augment_training_data(base_dir, file_list, output_file_name):
    num_images = len(file_list)
    augmented_array = np.zeros((num_images, num_rows, num_cols, num_channels))

    for (i, file_name) in enumerate(file_list):
        print(i, os.path.join(base_dir, file_name))
        img = cv2.cvtColor(cv2.imread(os.path.join(base_dir, file_name)), cv2.COLOR_BGR2GRAY)
        augmented_array[i, :, :, :] = cv2.resize(img, (num_rows, num_cols)).reshape(num_rows, num_cols, num_channels)
        cv2.imshow('test', augmented_array[i, :, :, :])
        k = cv2.waitKey(-1)
        if chr(k) == 'q':
            sys.exit(1)

    np.save(output_file_name, augmented_array)
    del augmented_array

base_dir = 'train/generated'
image_dir = os.path.join(base_dir, 'images')
mask_dir = os.path.join(base_dir, 'masks')
image_array_name = os.path.join(base_dir, 'augmented_images.npy')
mask_array_name = os.path.join(base_dir, 'augmented_masks.npy')
image_file_list = os.listdir(image_dir)
mask_file_list = os.listdir(mask_dir)
augment_training_data(image_dir, image_file_list, image_array_name)
augment_training_data(mask_dir, mask_file_list, mask_array_name)

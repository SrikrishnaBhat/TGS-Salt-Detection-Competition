import numpy as np
import os
import pandas as pd
import cv2

img_rows = 128
img_cols = 128
img_channels = 1
src_dir = 'train/generated'
dest_dir = os.path.join(src_dir, 'flipped_masks')
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
image_dir = os.path.join(src_dir, 'segmented_masks')

file_list = os.listdir(image_dir)

file_list.sort()

for (i, file_name) in enumerate(file_list):
    print(i, file_name)
    full_image_list = []
    image_array = np.load(os.path.join(image_dir, file_name))

    full_image_list.append(image_array)
    full_image_list.append(image_array[:, :, ::-1, :])
    full_image_list.append(image_array[:, ::-1, :, :])
    full_image_list.append(image_array[:, ::-1, ::-1, :])

    np.save(os.path.join(dest_dir, file_name), np.array(full_image_list).reshape(-1, img_rows, img_cols, img_channels))

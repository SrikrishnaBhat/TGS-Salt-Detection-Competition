import numpy as np
import os
import cv2

base_dir = 'train/generated'

dest_dir = os.path.join(base_dir, 'segmented_images')

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

src_file = os.path.join(base_dir, 'augmented_images.npy')
image_array = np.load(src_file)
image_chunk = 1000
image_shape = image_array.shape

j = 0

for i in range(0, image_shape[0], image_chunk):
    print(i)
    cv2.imshow('test', image_array[i])
    k=cv2.waitKey(10000)
    if chr(k) == 'q':
        break
    np.save(os.path.join(dest_dir, 'augmented_images_{}.npy'.format(j)), image_array[i:i+image_chunk, :, :, :])
    j += 1

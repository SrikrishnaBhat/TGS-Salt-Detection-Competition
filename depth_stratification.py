import numpy as np
import os

data_dict = np.load('resized_depth_based_train_data.npy').item()
X_train = data_dict['images']
Y_train = data_dict['masks']
depths = np.array(data_dict['depths'])

min_depth = 0
max_depth = 1000
range = 100

depth_partitions = np.arange(min_depth, max_depth, range)
dest_dir = 'stratified_vals'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

i = 0
while i<depth_partitions.shape[0]-1:
    file_name = os.path.join(dest_dir, '{}.npy'.format(i))
    flip_file_name = os.path.join(dest_dir, '{}_flip.npy'.format(i))
    print(i, file_name, flip_file_name)
    X_partition = X_train[(depth_partitions[i] <= depths) & (depths < depth_partitions[i+1])]
    Y_partition = Y_train[(depth_partitions[i] <= depths) & (depths < depth_partitions[i+1])]
    depth_part = depths[(depth_partitions[i] <= depths) & (depths < depth_partitions[i+1])]

    main_dict = {'images': X_partition, 'masks': Y_partition, 'depths': depth_part}
    np.save(file_name, main_dict)

    depth_flip_partition = np.array([depth_part, depth_part, depth_part]).flatten()
    new_shape_X = (-1,) + X_partition.shape[1:]
    new_shape_Y = (-1,) + Y_partition.shape[1:]
    X_flip_partition = np.array([np.fliplr(X_partition), np.flipud(X_partition), np.flip(X_partition)]).reshape(new_shape_X)
    Y_flip_partition = np.array([np.fliplr(Y_partition), np.flipud(Y_partition), np.flip(Y_partition)]).reshape(new_shape_Y)

    flip_dict = {'images': X_flip_partition, 'masks': Y_flip_partition, 'depths': depth_flip_partition}
    np.save(flip_file_name,  flip_dict)

    i += 1
import numpy as np
import sys

MODE_FULL = 'full'
MODE_LR = 'lr'
MODE_UD = 'ud'

def flip_lr(array):
    return array[:, ::-1, :]


def flip_uw(array):
    return array[::-1, :, :]


flip_dict = {
    MODE_FULL: np.flip,
    MODE_LR: flip_lr,
    MODE_UD: flip_uw
}

flip_mode_list = [MODE_LR, MODE_UD, MODE_FULL]

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python3 flip_image.py <flip_mode>')
        sys.exit(1)

    flip_mode = sys.argv[1]

    if flip_mode not in flip_mode_list:
        raise Exception('Expected flip modes: {}'.format(flip_mode_list))

    image_save_file_name = 'flipped_image_normal_{}.npy'.format(flip_mode)
    mask_save_file_name = 'flipped_mask_normal_{}.npy'.format(flip_mode)
    id_save_file_name = 'flipped_id_normal_{}.npy'.format(flip_mode)
    id_

    data_dict = np.load('resized_depth_based_train_data.npy').item()
    image_array =
    flipped_array = np.zeros_like(image_array)

    image_shape = image_array.shape
    flip_func = flip_dict[flip_mode]

    for i in range(image_shape[0]):
        print(i)
        flipped_array[i] = flip_func(image_array[i])

    np.save(save_file_name, flipped_array)
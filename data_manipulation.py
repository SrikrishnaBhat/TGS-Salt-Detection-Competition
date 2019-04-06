import numpy as np
import cv2
import os
from random import choice

def gen_mask_deformation(shape):
    total_len = np.prod(np.array(shape)).item()
    deform_index = np.random.permutation(np.array([i for i in range(total_len)]))
    return deform_index


def check_out_of_bounds(point, shape1, shape2):
    if (((point[0]-shape1[0]>=0) and (point[0] + shape1[0]<shape2[0])) and
            ((point[1]-shape1[1]>=0) and (point[1] + point[1]<shape2[1]))):
        return False
    return True


def group_gen(n):
    l = [i for i in range(n)]
    return np.random.permutation(l)


def apply_mask(mat, mask, shape, pos):
    window = mat[pos[0]-int(shape[0]/2):pos[0]+int(shape[0]/2), pos[1]-int(shape[1]/2):pos[1]+int(shape[1]/2)]
    mat[pos[0]-int(shape[0]/2):pos[0]+int(shape[0]/2), pos[1]-int(shape[1]/2):pos[1]+int(shape[1]/2)] = window.flatten()[mask].reshape(shape)


def gen_resized_deformation(image, gt, mask_centres, mask_shape, new_shape):
    for centre in mask_centres:
        # Get the indices to change
        start_x, start_y = centre[0] - mask_shape[0], centre[1] - mask_shape[0]
        end_x, end_y = centre[0] + mask_shape[0] + 1, centre[1] + mask_shape[1] + 1

        # Get the original patch
        orig_patch = image[start_x:end_x, start_y:end_y]
        orig_gt = gt[start_x:end_x, start_y:end_y]

        # Create the custom patch
        new_patch = cv2.resize(orig_patch, tuple((np.array(new_shape)*2 + 1).tolist()))
        new_gt = cv2.resize(orig_gt, tuple((np.array(new_shape)*2 + 1).tolist()))
        start_x, start_y = centre[0] - new_shape[0], centre[1] - new_shape[0]
        end_x, end_y = centre[0] + new_shape[0] + 1, centre[1] + new_shape[1] + 1

        # add the patches to matrices
        image[start_x:end_x, start_y:end_y] = new_patch
        gt[start_x:end_x, start_y:end_y] = new_gt

    return image, gt


def gen_deformed_matrix(image, gt, contours, deform_count, mask_shape):
    contour_inds = group_gen(deform_count)
    out_im = image.copy()
    out_gt = gt.copy()
    print(contour_inds)
    for ind in contour_inds:
        deform_pos = gen_mask_deformation(mask_shape)
        apply_mask(out_im, deform_pos, mask_shape, contours[ind])
        apply_mask(out_gt, deform_pos, mask_shape, contours[ind])

    return out_im, out_gt


def get_deformed_matrices(image, gt, mask_shape, new_shape, deform_count, group_count=1):
    image_shape = image.shape
    #ref_shape = (image_shape[0] - new_shape[1], image_shape[1] - new_shape[1])
    max_shape = (max(new_shape[0], mask_shape[0]), max(new_shape[1], mask_shape[1]))
    ref_array_x = np.arange(max_shape[0], max_shape[0] - max_shape[0])
    ref_array_y = np.arange(max_shape[1], max_shape[1] - max_shape[1])
    im_list, gt_list = [], []

    for i in range(group_count):
        np.random.shuffle(ref_array_x)
        np.random.shuffle(ref_array_y)
        centres = np.transpose(np.array([ref_array_x[:deform_count], ref_array_y[:deform_count]]))
        out_im, out_gt = gen_resized_deformation(image.copy(), gt.copy(), centres, mask_shape, new_shape)
        im_list.append(out_im)
        gt_list.append(out_gt)

    return im_list, gt_list


base_dir = 'train'
image_dir = os.path.join(base_dir, 'images')
mask_dir = os.path.join(base_dir, 'masks')
file_gen_dir = os.path.join(base_dir, 'generated')
gen_image_dir = os.path.join(file_gen_dir, 'images')
gen_mask_dir = os.path.join(file_gen_dir, 'masks')

if not os.path.exists(file_gen_dir):
    os.makedirs(file_gen_dir)
if not os.path.exists(gen_image_dir):
    os.makedirs(gen_image_dir)
if not os.path.exists(gen_mask_dir):
    os.makedirs(gen_mask_dir)


im_file_list = os.listdir(image_dir)
mask_file_list = os.listdir(mask_dir)
size_list = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]


deformed_count = 10
group_count = 10


for i, image_file in enumerate(mask_file_list):
    print(i, image_file)
    name, ext = os.path.splitext(image_file)
    image = cv2.cvtColor(cv2.imread(os.path.join(image_dir, image_file)), cv2.COLOR_BGR2GRAY)
    mask = cv2.cvtColor(cv2.imread(os.path.join(mask_dir, image_file)), cv2.COLOR_BGR2GRAY)
    mask_shape = choice(size_list)
    size_list.remove(mask_shape)
    new_shape = choice(size_list)
    size_list.append(mask_shape)
    im_list, gt_list = get_deformed_matrices(image, mask, mask_shape, new_shape, deformed_count, group_count)

    for j in range(len(im_list)):
        cv2.imwrite(os.path.join(gen_image_dir, '{}_{}.png'.format(name, j)), im_list[j])
        cv2.imwrite(os.path.join(gen_mask_dir, '{}_{}.png'.format(name, j)), gt_list[j])

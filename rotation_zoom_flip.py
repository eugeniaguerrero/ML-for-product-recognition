import numpy as np
import scipy.ndimage as ndi

def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.):
    theta = np.deg2rad(np.random.uniform(-rg, rg))
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)

    transformed_tensors = []
    for i in range(x.shape[0]):
        single_tensor = apply_transform(x[i], transform_matrix, channel_axis, fill_mode, cval)
        transformed_tensors.append(single_tensor)
    transformed_tensors = np.array(transformed_tensors)

    return transformed_tensors

def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two floats. ''Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1

    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transformed_tensors = []
        h, w = x[0].shape[row_axis], x[0].shape[col_axis]
        transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
            
        for i in range(x.shape[0]):
            single_tensor = apply_transform(x[i], transform_matrix, channel_axis, fill_mode, cval)
            transformed_tensors.append(single_tensor)

        transformed_tensors = np.array(transformed_tensors)
                
        return transformed_tensors

def flip_horizontal(x):
    transformed_tensors = []
    for i in range(x.shape[0]):
        single_tensor = flip_axis(x[i], 1)
        transformed_tensors.append(single_tensor)

    transformed_tensors = np.array(transformed_tensors)
    return transformed_tensors

def flip_vertical(x):
    transformed_tensors = []
    for i in range(x.shape[0]):
        single_tensor = flip_axis(x[i], 0)
        transformed_tensors.append(single_tensor)

    transformed_tensors = np.array(transformed_tensors)
    return transformed_tensors

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
    
def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix, final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
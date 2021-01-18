import numpy as np

num_patch = 4

def get_square(img, pos):
    """
    Get one patch of the image based on position
    Arg:
      img: numpy array
      pos: tuple, shape_like = (row, column)
    Returns:
      a patch
    """
    h = img.shape[0]
    w = img.shape[1]
    h_patch = int(h / num_patch)
    w_patch = int(w / num_patch)
    return img[pos[0]*h_patch:(pos[0]+1)*h_patch, pos[1]*w_patch:(pos[1]+1)*w_patch]

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

# def normalize(x):
#     x = x.astype('float64')
#     x -= x.mean(axis=(1, 2), keepdims=True)
#     x /= x.std(axis=(1, 2), keepdims=True)
#     return x

def normalize(x):
    return x/255


def merge_masks(prob_list, image_shape):
    """
    Merge patches of masks
    ----
    Arg:
      prob_list:
      image_shape: tuple, size=(h,w)
    """
    new = np.zeros(image_shape, np.float32)
    h_patch, w_patch = int(image_shape[0]/num_patch), int(image_shape[1]/num_patch)
    counter = 0
    for i in range(num_patch):
        for j in range(num_patch):
            new[i*h_patch:(i+1)*h_patch, j*w_patch:(j+1)*w_patch] = prob_list[counter]
            counter+=1
    return new


def get_row_batches(img, mask, num_patch):
    """ Divide 2048 by 2048 raw image into a list of row patches
    Arg:
    ------
    img: shape=(c, h, w)
    """

    h = img.shape[1]
    w = img.shape[2]
    h_patch = int(h / num_patch[0])
    w_patch = int(w / num_patch[1])
    return (
        np.array([[img[:, i*h_patch:(i+1)*h_patch, j*w_patch:(j+1)*w_patch] for j in range(num_patch[1])] for i in range(num_patch[0])]),
        np.array([[mask[i*h_patch:(i+1)*h_patch, j*w_patch:(j+1)*w_patch] for j in range(num_patch[1])] for i in range(num_patch[0])])
           )

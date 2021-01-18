import sys
import numpy as np
from itertools import product
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from Utils.transform import *
from Utils.pillowhelper import *


def rowcolumn2coor(row, col, patch_size):
    """ Map row column number to pillow image coordinates: (left, upper, right, lower)
    """
    left = col * patch_size
    upper = row * patch_size
    right = (col + 1) * patch_size
    lower = (row + 1) * patch_size
    return (left, upper, right, lower)


def main_road_knn(image, feature_extractor, z_list, thresh, num_patch, patch_ignore, patch_size, flag_equalizer=True, img_resize=224, flag_cuda=True):
    z_list_test = get_latent_vector_list_test(image, feature_extractor, num_patch, patch_ignore, patch_size, flag_equalizer, img_resize, flag_cuda)
    detected = detect_anomaly_knn(z_list, z_list_test, thresh, num_patch, patch_ignore)
    return detected


def get_latent_vector_list_test(image, feature_extractor, num_patch, patch_ignore, patch_size, flag_equalizer, img_resize, flag_cuda):    
    # Extraction
    z_list_patches = []
    for i, (row, col) in enumerate(product(range(num_patch), range(num_patch))):
        if patch_ignore and (row, col) in patch_ignore:
            print('skip {}'.format((row, col)))
            continue
        print('compute {}'.format((row, col)))
        tmp_coor = rowcolumn2coor(row, col, patch_size)
        # Apply transformer
        tmp_transforms = transforms.Compose([
            EqualizerCroppedGrey(flag_equalizer, tmp_coor, img_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            RepeatChannel(3)
        ])
        tmp_patch = tmp_transforms(image)
        tmp_patch = tmp_patch.unsqueeze(0)
        if flag_cuda:
            tmp_patch = tmp_patch.cuda()
        tmp_z = feature_extractor(tmp_patch).detach().cpu().numpy()
        z_list_patches.append(tmp_z)
        
    tmp_transforms = transforms.Compose([
        EqualizerCroppedGrey(flag_equalizer, None, img_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        RepeatChannel(3)
    ])

    tmp_patch = tmp_transforms(image)
    tmp_patch = tmp_patch.unsqueeze(0)
    if flag_cuda:
        tmp_patch = tmp_patch.cuda()
    tmp_z = feature_extractor(tmp_patch).detach().cpu().numpy()
    z_list_patches.append(tmp_z)
    return z_list_patches


def detect_anomaly_knn(z_list, z_list_test, thresh, num_patch, patch_ignore):
    counter = 0
    detected = []
    for i, (row, col) in enumerate(product(range(num_patch), range(num_patch))):
        if patch_ignore and (row, col) in patch_ignore:
            print('skip {}'.format((row, col)))
            continue
        print('compute {}'.format((row, col)))
        score = np.mean(cosine_similarity(z_list[counter], z_list_test[counter]), axis=0)
        if score[0] < thresh[counter]:
            detected.append({'index': counter, 'row': row, 'col': col, 'score': score[0]})
        counter+=1
    score = np.mean(cosine_similarity(z_list[counter], z_list_test[counter]), axis=0)
    if score[0] < thresh[counter]:
        detected.append({'index': counter, 'row': None, 'col': None, 'score': score[0]})
    return detected
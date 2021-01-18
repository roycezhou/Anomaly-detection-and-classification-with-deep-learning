import os
dir = os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], ".."))
wd_dir =dir+'/src'


import os
import sys
import shutil
import base64
from collections import OrderedDict
import time
import pytz
from datetime import datetime
from pytz import timezone
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.append(wd_dir)
from unet.unet import UNet
from unet.util_image import get_square, hwc_to_chw, normalize

from collections import OrderedDict
from PIL import Image, ImageOps


from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import time
from functools import partial



def rowcolumn2coor(row, col, patch_size):
    """ Map row column number to pillow image coordinates: (left, upper, right, lower)
    """
    left = col * patch_size
    upper = row * patch_size
    right = (col + 1) * patch_size
    lower = (row + 1) * patch_size
    return (left, upper, right, lower)


def merge_masks(prob_list):
    """ merge the output of different patch together
        """
    h_raw, w_raw = 2048, 2048
    num_patch = 4
    new = np.zeros((h_raw, w_raw), np.float32)
    h_patch, w_patch = int(h_raw / num_patch), int(w_raw / num_patch)
    counter = 0
    for i in range(num_patch):
        for j in range(num_patch):
            new[i * h_patch:(i + 1) * h_patch, j * w_patch:(j + 1) * w_patch] = prob_list[counter]
            counter += 1
    return new


def mask_to_image(mask):
    """ change the mask from numpy array to image
            """
    return Image.fromarray((mask * 255).astype(np.uint8))


def imageprepare(text):
    """ prepare the image, convert from string to gray channel image array
             Args:
             ------
                 text: the input of base64 image
                """
    nparr = np.fromstring(base64.b64decode(text), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return RGB


def decodeParallel(patch_list, threads=2):
    """ initiate a pool to control multithread
             Args:
             ------
                 patch_list: list of input patch array
                 thread: the number of thread you want to run concurrently
                   """
    pool = ThreadPool(threads)
    # map will take the every item in patch_list to apply the decode funciton
    new_list = pool.map(partial(decode), patch_list)
    pool.close()
    pool.join()
    return new_list



def decode(imgs):
    """ the function run in the multithread pool
            Args:
            ------
                imgs: a single input from the patch_list in the pool
                      """
    import subprocess
    from io import BytesIO
    import os
    dir = os.path.split(os.path.realpath(__file__))[0]
    decode_file = dir + '/application_decode.py'
    buffered = BytesIO()
    tmp_pos, patch, app = imgs
    if(app==b'Edge'):
        patch = patch.astype('float')
        patchstring = patch.tostring()
        process = subprocess.run(['/anaconda_env/personal/myproject/DLDtorch1/bin/python',
                                  decode_file],
                                 input=base64.b64encode(patchstring) + app, stdout=subprocess.PIPE)
        decode_output = process.stdout[2:-2]
    else:
        patch.save(buffered, format="JPEG")
        patch_str = buffered.getvalue()
        process = subprocess.run(['/anaconda_env/personal/myproject/DLDtorch1/bin/python',
                                  decode_file],
                                 input=base64.b64encode(patch_str) + app, stdout=subprocess.PIPE)
        decode_output = process.stdout[2:-2]
    img_data = base64.b64decode(decode_output)
    nparr = np.frombuffer(img_data, float)
    output_array=nparr.reshape(1,512,512)
    return [tmp_pos, output_array]


def MasklossScoring(image_base64, image_name,bin_thresh=0.2, thresh_num_pixel=1,
                      flag_mask_edge=False, flag_mask_inner=False, flag_equalizer=False):
    """ Predict if the pillow image have Streak
    """
    # set the list to crop the image into 4x4, 16 batches

    imgtext = image_base64 + '==='
    imgRGB = imageprepare(imgtext)
    img_pillow = Image.fromarray(imgRGB.astype('uint8')).convert('RGB')

    num_patch = 4
    patch_size = 512
    app = b'Mask'

    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    # radius = 920
    radius_inner = 920
    radius_outer = 960

    y_grid, x_grid = np.ogrid[-a_center:img_size_mask - a_center, -b_center:img_size_mask - b_center]
    mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_outer * radius_outer
    if flag_mask_edge:
        mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_inner * radius_inner
    elif flag_mask_inner:
        mask_ignore = np.logical_or(mask_ignore, (x_grid * x_grid + y_grid * y_grid <= radius_inner * radius_inner))

    pos = [(i, j) for i in range(num_patch) for j in range(num_patch)]
    if flag_equalizer:
        RGB_image = ImageOps.equalize(img_pillow)
    else:
        RGB_image = img_pillow

    imgs_patchs = []
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_image.crop(coor_pil)
        imgs_patchs.append((tmp_pos,tmp_image_patch,app))

    tic = time.time()
    appruntime = datetime.now(pytz.timezone('Singapore')).strftime('%Y-%m-%d %H:%M:%S')
    print('start decode Parallel:' + appruntime)


    output_patch = decodeParallel(imgs_patchs, 4)
    output_patch_list = list(output_patch)
    output_patch_list.sort()
    prob_list = []
    for i, patch in output_patch_list:
        prob_list.append(patch)
    full_mask = merge_masks(prob_list).astype(np.float32)

    toc = time.time()
    print('the mask finish time:' + str(app) + str(toc - tic))

    full_mask[mask_ignore] = 0
    full_mask_bin = full_mask > bin_thresh
    num_positive_pixel = np.sum(full_mask_bin)

    if num_positive_pixel > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False

    return label, base64.b64encode(full_mask)



def EdgeArcingScoring (image_base64, imagename, bin_thresh=0.2, thresh_num_pixel=1, flag_mask_edge=False):
    """ Predict if the pillow image have EdgeArcing
    """
    # set the list to crop the image into 4x4, 16 batches
    num_patch = 4
    pos = [(i, j) for i in range(num_patch) for j in range(num_patch)]
    app=b'Edge'
    #get the level and step info from the imagename, different level have different oarameter setting
    step = imagename.split('::')[1]
    level = imagename.split('::')[2]
    if level == '03':
        bin_thresh = 0.15
        num_pixel_thresh = 5
        brightness_thresh = 100
    else:
        bin_thresh = 0.1
        num_pixel_thresh = 5

    #prepare the image changing from base64 to image array, and only take one channel
    imgtext = image_base64 + '==='
    imgRGB = imageprepare(imgtext)
    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
    imgarr = np.expand_dims(np.dot(imgBGR[..., :3], [0.2989, 0.5870, 0.1140]), axis=2)
    imgarr = normalize(imgarr)

    # crop to image
    imgs_patchs = [(tmp_pos, hwc_to_chw(get_square(imgarr, tmp_pos)), app) for tmp_pos in pos]
    tic = time.time()
    appruntime = datetime.now(pytz.timezone('Singapore')).strftime('%Y-%m-%d %H:%M:%S')
    print('start decode Parallel:' + appruntime)

    # call the decode parallel to set the decode pool and get the output in order
    output_patch = decodeParallel(imgs_patchs, 4)
    output_patch_list = list(output_patch)
    output_patch_list.sort()
    prob_list = []
    for i, patch in output_patch_list:
        prob_list.append(patch)
    full_mask = merge_masks(prob_list).astype(np.float32)
    toc = time.time()
    print('the mask finish time:' + str(app) + str(toc - tic))

    # Modify the full mask so that the probability of non-edge region is zero
    a_center, b_center = 1024, 1024
    img_size_mask = 2048
    radius = 920
    y_grid, x_grid = np.ogrid[-a_center:img_size_mask - a_center, -b_center:img_size_mask - b_center]
    mask_non_edge = x_grid * x_grid + y_grid * y_grid <= radius * radius
    if flag_mask_edge:
        full_mask[mask_non_edge] = 0


    # analysis the output to get the label
    full_mask_bin = full_mask > bin_thresh
    num_positive_pixel = np.sum(full_mask_bin)
    flag_positive = (full_mask.max() > bin_thresh) and (num_positive_pixel > num_pixel_thresh)
    if (level == '03') and flag_positive:
        detected_pixels = imgRGB[full_mask_bin]
        brightness = 0.2126 * detected_pixels[:, 0] + 0.7152 * detected_pixels[:, 1] + 0.0722 * detected_pixels[:, 2]
        if brightness.mean() > brightness_thresh:
            flag_positive = False

    return flag_positive, base64.b64encode(full_mask)



def DefocusScoring(image_base64, image_name,bin_thresh=0.2, thresh_num_pixel=1,
                      flag_mask_edge=False, flag_mask_inner=False, flag_equalizer=False):
    """ Predict if the pillow image have Streak
    """
    # set the list to crop the image into 4x4, 16 batches

    imgtext = image_base64 + '==='
    imgRGB = imageprepare(imgtext)
    img_pillow = Image.fromarray(imgRGB.astype('uint8')).convert('RGB')

    num_patch = 4
    patch_size = 512
    app = b'Defo'

    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    # radius = 920
    radius_inner = 920
    radius_outer = 960

    y_grid, x_grid = np.ogrid[-a_center:img_size_mask - a_center, -b_center:img_size_mask - b_center]
    mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_outer * radius_outer
    if flag_mask_edge:
        mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_inner * radius_inner
    elif flag_mask_inner:
        mask_ignore = np.logical_or(mask_ignore, (x_grid * x_grid + y_grid * y_grid <= radius_inner * radius_inner))

    pos = [(i, j) for i in range(num_patch) for j in range(num_patch)]
    if flag_equalizer:
        RGB_image = ImageOps.equalize(img_pillow)
    else:
        RGB_image = img_pillow

    imgs_patchs = []
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_image.crop(coor_pil)
        imgs_patchs.append((tmp_pos,tmp_image_patch,app))

    tic = time.time()
    appruntime = datetime.now(pytz.timezone('Singapore')).strftime('%Y-%m-%d %H:%M:%S')
    print('start decode Parallel:' + appruntime)


    output_patch = decodeParallel(imgs_patchs, 4)
    output_patch_list = list(output_patch)
    output_patch_list.sort()
    prob_list = []
    for i, patch in output_patch_list:
        prob_list.append(patch)
    full_mask = merge_masks(prob_list).astype(np.float32)

    toc = time.time()
    print('the mask finish time:' + str(app) + str(toc - tic))

    full_mask[mask_ignore] = 0
    full_mask_bin = full_mask > bin_thresh
    num_positive_pixel = np.sum(full_mask_bin)

    if num_positive_pixel > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False

    return label, base64.b64encode(full_mask)


def CometScoring(image_base64, image_name,bin_thresh=0.2, thresh_num_pixel=1,
                      flag_mask_edge=False, flag_mask_inner=False, flag_equalizer=False):
    """ Predict if the pillow image have Streak
    """
    # set the list to crop the image into 4x4, 16 batches

    imgtext = image_base64 + '==='
    imgRGB = imageprepare(imgtext)
    img_pillow = Image.fromarray(imgRGB.astype('uint8')).convert('RGB')

    num_patch = 4
    patch_size = 512
    app = b'Come'

    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    # radius = 920
    radius_inner = 920
    radius_outer = 960

    y_grid, x_grid = np.ogrid[-a_center:img_size_mask - a_center, -b_center:img_size_mask - b_center]
    mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_outer * radius_outer
    if flag_mask_edge:
        mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_inner * radius_inner
    elif flag_mask_inner:
        mask_ignore = np.logical_or(mask_ignore, (x_grid * x_grid + y_grid * y_grid <= radius_inner * radius_inner))

    pos = [(i, j) for i in range(num_patch) for j in range(num_patch)]
    if flag_equalizer:
        RGB_image = ImageOps.equalize(img_pillow)
    else:
        RGB_image = img_pillow

    imgs_patchs = []
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_image.crop(coor_pil)
        imgs_patchs.append((tmp_pos,tmp_image_patch,app))

    tic = time.time()
    appruntime = datetime.now(pytz.timezone('Singapore')).strftime('%Y-%m-%d %H:%M:%S')
    print('start decode Parallel:' + appruntime)


    output_patch = decodeParallel(imgs_patchs, 4)
    output_patch_list = list(output_patch)
    output_patch_list.sort()
    prob_list = []
    for i, patch in output_patch_list:
        prob_list.append(patch)
    full_mask = merge_masks(prob_list).astype(np.float32)

    toc = time.time()
    print('the mask finish time:' + str(app) + str(toc - tic))

    full_mask[mask_ignore] = 0
    full_mask_bin = full_mask > bin_thresh
    num_positive_pixel = np.sum(full_mask_bin)

    if num_positive_pixel > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False

    return label, base64.b64encode(full_mask)

def SmudgeScoring(image_base64, image_name,bin_thresh=0.2, thresh_num_pixel=1,
                      flag_mask_edge=False, flag_mask_inner=False, flag_equalizer=False):
    """ Predict if the pillow image have Streak
    """
    # set the list to crop the image into 4x4, 16 batches

    imgtext = image_base64 + '==='
    imgRGB = imageprepare(imgtext)
    img_pillow = Image.fromarray(imgRGB.astype('uint8')).convert('RGB')

    num_patch = 4
    patch_size = 512
    app = b'Smud'

    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    # radius = 920
    radius_inner = 920
    radius_outer = 960

    y_grid, x_grid = np.ogrid[-a_center:img_size_mask - a_center, -b_center:img_size_mask - b_center]
    mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_outer * radius_outer
    if flag_mask_edge:
        mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_inner * radius_inner
    elif flag_mask_inner:
        mask_ignore = np.logical_or(mask_ignore, (x_grid * x_grid + y_grid * y_grid <= radius_inner * radius_inner))

    pos = [(i, j) for i in range(num_patch) for j in range(num_patch)]
    if flag_equalizer:
        RGB_image = ImageOps.equalize(img_pillow)
    else:
        RGB_image = img_pillow

    imgs_patchs = []
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_image.crop(coor_pil)
        imgs_patchs.append((tmp_pos,tmp_image_patch,app))

    tic = time.time()
    appruntime = datetime.now(pytz.timezone('Singapore')).strftime('%Y-%m-%d %H:%M:%S')
    print('start decode Parallel:' + appruntime)


    output_patch = decodeParallel(imgs_patchs, 4)
    output_patch_list = list(output_patch)
    output_patch_list.sort()
    prob_list = []
    for i, patch in output_patch_list:
        prob_list.append(patch)

    full_mask = merge_masks(prob_list).astype(np.float32)


    toc = time.time()
    print('the mask finish time:' + str(app) + str(toc - tic))

    full_mask[mask_ignore] = 0
    full_mask_bin = full_mask > bin_thresh
    num_positive_pixel = np.sum(full_mask_bin)

    if num_positive_pixel > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False

    return label, base64.b64encode(full_mask)

def CactusScoring(image_base64, image_name,bin_thresh=0.2, thresh_num_pixel=1,
                      flag_mask_edge=False, flag_mask_inner=False, flag_equalizer=False):
    """ Predict if the pillow image have Streak
    """
    # set the list to crop the image into 4x4, 16 batches

    imgtext = image_base64 + '==='
    imgRGB = imageprepare(imgtext)
    img_pillow = Image.fromarray(imgRGB.astype('uint8')).convert('RGB')

    num_patch = 4
    patch_size = 512
    app = b'Cact'

    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    # radius = 920
    radius_inner = 920
    radius_outer = 960

    y_grid, x_grid = np.ogrid[-a_center:img_size_mask - a_center, -b_center:img_size_mask - b_center]
    mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_outer * radius_outer
    if flag_mask_edge:
        mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_inner * radius_inner
    elif flag_mask_inner:
        mask_ignore = np.logical_or(mask_ignore, (x_grid * x_grid + y_grid * y_grid <= radius_inner * radius_inner))

    pos = [(i, j) for i in range(num_patch) for j in range(num_patch)]
    if flag_equalizer:
        RGB_image = ImageOps.equalize(img_pillow)
    else:
        RGB_image = img_pillow

    imgs_patchs = []
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_image.crop(coor_pil)
        imgs_patchs.append((tmp_pos,tmp_image_patch,app))

    tic = time.time()
    appruntime = datetime.now(pytz.timezone('Singapore')).strftime('%Y-%m-%d %H:%M:%S')
    print('start decode Parallel:' + appruntime)


    output_patch = decodeParallel(imgs_patchs, 4)
    output_patch_list = list(output_patch)
    output_patch_list.sort()
    prob_list = []
    for i, patch in output_patch_list:
        prob_list.append(patch)
    full_mask = merge_masks(prob_list).astype(np.float32)

    toc = time.time()
    print('the mask finish time:' + str(app) + str(toc - tic))

    full_mask[mask_ignore] = 0
    full_mask_bin = full_mask > bin_thresh
    num_positive_pixel = np.sum(full_mask_bin)

    if num_positive_pixel > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False

    return label, base64.b64encode(full_mask)

def StreakScoring(image_base64, image_name,bin_thresh=0.2, thresh_num_pixel=1,
                      flag_mask_edge=False, flag_mask_inner=False, flag_equalizer=False):
    """ Predict if the pillow image have Streak
    """
    # set the list to crop the image into 4x4, 16 batches

    imgtext = image_base64 + '==='
    imgRGB = imageprepare(imgtext)
    img_pillow = Image.fromarray(imgRGB.astype('uint8')).convert('RGB')

    num_patch = 4
    patch_size = 512
    app = b'Stre'

    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    # radius = 920
    radius_inner = 920
    radius_outer = 960

    y_grid, x_grid = np.ogrid[-a_center:img_size_mask - a_center, -b_center:img_size_mask - b_center]
    mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_outer * radius_outer
    if flag_mask_edge:
        mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_inner * radius_inner
    elif flag_mask_inner:
        mask_ignore = np.logical_or(mask_ignore, (x_grid * x_grid + y_grid * y_grid <= radius_inner * radius_inner))

    pos = [(i, j) for i in range(num_patch) for j in range(num_patch)]
    if flag_equalizer:
        RGB_image = ImageOps.equalize(img_pillow)
    else:
        RGB_image = img_pillow

    imgs_patchs = []
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_image.crop(coor_pil)
        imgs_patchs.append((tmp_pos,tmp_image_patch,app))

    tic = time.time()
    appruntime = datetime.now(pytz.timezone('Singapore')).strftime('%Y-%m-%d %H:%M:%S')
    print('start decode Parallel:' + appruntime)


    output_patch = decodeParallel(imgs_patchs, 4)
    output_patch_list = list(output_patch)
    output_patch_list.sort()
    prob_list = []
    for i, patch in output_patch_list:
        prob_list.append(patch)
    full_mask = merge_masks(prob_list).astype(np.float32)

    toc = time.time()
    print('the mask finish time:' + str(app) + str(toc - tic))

    full_mask[mask_ignore] = 0
    full_mask_bin = full_mask > bin_thresh
    num_positive_pixel = np.sum(full_mask_bin)

    if num_positive_pixel > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False

    return label, base64.b64encode(full_mask)






def MScratchScoring(image_base64, image_name,bin_thresh=0.2, thresh_num_pixel=1,
                      flag_mask_edge=True, flag_mask_inner=False, flag_equalizer=True):
    """ Predict if the pillow image have Streak
    """
    # set the list to crop the image into 4x4, 16 batches

    imgtext = image_base64 + '==='
    imgRGB = imageprepare(imgtext)
    img_pillow = Image.fromarray(imgRGB.astype('uint8')).convert('RGB')

    num_patch = 4
    patch_size = 512
    app = b'Scra'

    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    # radius = 920
    radius_inner = 920
    radius_outer = 960

    y_grid, x_grid = np.ogrid[-a_center:img_size_mask - a_center, -b_center:img_size_mask - b_center]
    mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_outer * radius_outer
    if flag_mask_edge:
        mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_inner * radius_inner
    elif flag_mask_inner:
        mask_ignore = np.logical_or(mask_ignore, (x_grid * x_grid + y_grid * y_grid <= radius_inner * radius_inner))

    pos = [(i, j) for i in range(num_patch) for j in range(num_patch)]
    if flag_equalizer:
        RGB_image = ImageOps.equalize(img_pillow)
    else:
        RGB_image = img_pillow

    imgs_patchs = []
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_image.crop(coor_pil)
        imgs_patchs.append((tmp_pos,tmp_image_patch,app))

    tic = time.time()
    appruntime = datetime.now(pytz.timezone('Singapore')).strftime('%Y-%m-%d %H:%M:%S')
    print('start decode Parallel:' + appruntime)


    output_patch = decodeParallel(imgs_patchs, 4)
    output_patch_list = list(output_patch)
    output_patch_list.sort()
    prob_list = []
    for i, patch in output_patch_list:
        prob_list.append(patch)
    full_mask = merge_masks(prob_list).astype(np.float32)

    toc = time.time()
    print('the mask finish time:' + str(app) + str(toc - tic))

    full_mask[mask_ignore] = 0
    full_mask_bin = full_mask > bin_thresh
    num_positive_pixel = np.sum(full_mask_bin)

    if num_positive_pixel > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False

    return label, base64.b64encode(full_mask)


def MaskLossOrg(net, img_pillow, num_patch, patch_size, bin_thresh=0.2, thresh_num_pixel=1,
                      flag_mask_edge=False, flag_mask_inner=False, flag_equalizer=False):

    #     print('{}  Start'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    # radius = 920
    radius_inner = 920
    radius_outer = 960

    import torch
    import torchvision.transforms as transforms
    from Utils.pillowhelper import rowcolumn2coor

    y_grid, x_grid = np.ogrid[-a_center:img_size_mask - a_center, -b_center:img_size_mask - b_center]
    mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_outer * radius_outer
    if flag_mask_edge:
        mask_ignore = x_grid * x_grid + y_grid * y_grid >= radius_inner * radius_inner
    elif flag_mask_inner:
        mask_ignore = np.logical_or(mask_ignore, (x_grid * x_grid + y_grid * y_grid <= radius_inner * radius_inner))

    pos = [(i, j) for i in range(num_patch) for j in range(num_patch)]
    if flag_equalizer:
        RGB_image = ImageOps.equalize(img_pillow)
    else:
        RGB_image = img_pillow

    prob_list = []
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_image.crop(coor_pil)
        tmp_image_patch_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])(np.array(tmp_image_patch))

        tmp_data = tmp_image_patch_tensor.type(torch.FloatTensor).unsqueeze(0)
        with torch.no_grad():
            tmp_output = net(tmp_data).squeeze(0).cpu().numpy()
            prob_list.append(tmp_output)

    full_mask = merge_masks(prob_list)

    full_mask[mask_ignore] = 0
    full_mask_bin = full_mask > bin_thresh
    num_positive_pixel = np.sum(full_mask_bin)

    if num_positive_pixel > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False
    #     print('{}  finished'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    return label, num_positive_pixel, full_mask
#eadge arching generate not include yet

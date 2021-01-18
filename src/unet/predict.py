import sys
import numpy as np
import datetime
from collections import OrderedDict
from PIL import Image, ImageOps

import torch
import torchvision.transforms as transforms
from Utils.pillowhelper import rowcolumn2coor


def merge_masks(prob_list, num_patch, h_raw, w_raw):
        new = np.zeros((h_raw, w_raw), np.float32)
        h_patch, w_patch = int(h_raw/num_patch), int(w_raw/num_patch)
        counter = 0
        for i in range(num_patch):
            for j in range(num_patch):
                new[i*h_patch:(i+1)*h_patch, j*w_patch:(j+1)*w_patch] = prob_list[counter]
                counter+=1
        return new


def load_state_dict_from_multigpu2cpu(state_dir):
    """ To load weights trained using multiple gpu to cpu
    """
    multigpu_state_dict = torch.load(state_dir, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in multigpu_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def predict_photo_defocus(net, img_pillow, num_patch, patch_size, bin_thresh, thresh_num_pixel=1, flag_mask_edge=True):
    """ Predict if the pillow image have defocus
    Args:
    ------
      net: pytorch net with cpu device
    """
    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    #radius = 920
    radius = 910
    y_grid,x_grid = np.ogrid[-a_center:img_size_mask-a_center, -b_center:img_size_mask-b_center]
    mask_edge = x_grid*x_grid + y_grid*y_grid >= radius*radius
    
    pos = [(i,j) for i in range(num_patch) for j in range(num_patch)]
    RGB_equalizer = ImageOps.equalize(img_pillow)
    
    prob_list = []
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_equalizer.crop(coor_pil)
        tmp_image_patch_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])(tmp_image_patch)
        
        tmp_data = tmp_image_patch_tensor.type(torch.FloatTensor).unsqueeze(0)
        with torch.no_grad():
            tmp_output = net(tmp_data).squeeze(0).cpu().numpy()
            prob_list.append(tmp_output)
            
    full_mask = merge_masks(prob_list, num_patch, img_pillow.size[0], img_pillow.size[1])
    
    if flag_mask_edge:
        full_mask[mask_edge] = 0
    full_mask_bin = full_mask > bin_thresh
    
    if np.sum(full_mask_bin) > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False
    return label, full_mask


def predict_photo_comet(net, img_pillow, num_patch, patch_size, bin_thresh, thresh_num_pixel=1, flag_mask_edge=True):
    """ Predict if the pillow image have the defect under investigation
    Args:
    ------
      net: pytorch net with cpu device
    """
    print('{}  Start'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    #radius = 920
    radius = 910
    y_grid,x_grid = np.ogrid[-a_center:img_size_mask-a_center, -b_center:img_size_mask-b_center]
    mask_edge = x_grid*x_grid + y_grid*y_grid >= radius*radius
    
    pos = [(i,j) for i in range(num_patch) for j in range(num_patch)]
    RGB_equalizer = ImageOps.equalize(img_pillow)
    
    
    prob_list = []
    print('{}  predicting patches'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_equalizer.crop(coor_pil)
        tmp_image_patch_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])(tmp_image_patch)
        
        tmp_data = tmp_image_patch_tensor.type(torch.FloatTensor).unsqueeze(0)
        with torch.no_grad():
            tmp_output = net(tmp_data).squeeze(0).cpu().numpy()
            prob_list.append(tmp_output)
        print('{}  patch'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
            
    full_mask = merge_masks(prob_list, num_patch, img_pillow.size[0], img_pillow.size[1])
    
    if flag_mask_edge:
        full_mask[mask_edge] = 0
    full_mask_bin = full_mask > bin_thresh
    
    if np.sum(full_mask_bin) > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False
    print('{}  finished'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    return label, full_mask, RGB_equalizer


def predict_photo_smudge(net, img_pillow, num_patch, patch_size, bin_thresh=0.2, thresh_num_pixel=1, flag_mask_edge=True):
    """ Predict if the pillow image have the defect under investigation
    Args:
    ------
      net: pytorch net with cpu device
    """
    print('{}  Start'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    #radius = 920
    radius = 910
    y_grid,x_grid = np.ogrid[-a_center:img_size_mask-a_center, -b_center:img_size_mask-b_center]
    mask_edge = x_grid*x_grid + y_grid*y_grid >= radius*radius
    
    pos = [(i,j) for i in range(num_patch) for j in range(num_patch)]
    RGB_equalizer = ImageOps.equalize(img_pillow)
    
    
    prob_list = []
    print('{}  predicting patches'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_equalizer.crop(coor_pil)
        tmp_image_patch_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])(tmp_image_patch)
        
        tmp_data = tmp_image_patch_tensor.type(torch.FloatTensor).unsqueeze(0)
        with torch.no_grad():
            tmp_output = net(tmp_data).squeeze(0).cpu().numpy()
            prob_list.append(tmp_output)
        print('{}  patch'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
            
    full_mask = merge_masks(prob_list, num_patch, img_pillow.size[0], img_pillow.size[1])
    
    if flag_mask_edge:
        full_mask[mask_edge] = 0
    full_mask_bin = full_mask > bin_thresh
    
    if np.sum(full_mask_bin) > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False
    print('{}  finished'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    return label, full_mask, RGB_equalizer


def predict_photo_cactus(net, img_pillow, num_patch, patch_size, bin_thresh=0.2, thresh_num_pixel=1, flag_mask_edge=True):
    """ Predict if the pillow image have the defect under investigation
    Args:
    ------
      net: pytorch net with cpu device
    """
    print('{}  Start'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    #radius = 920
    radius = 910
    y_grid,x_grid = np.ogrid[-a_center:img_size_mask-a_center, -b_center:img_size_mask-b_center]
    mask_edge = x_grid*x_grid + y_grid*y_grid >= radius*radius
    
    pos = [(i,j) for i in range(num_patch) for j in range(num_patch)]
    RGB_equalizer = ImageOps.equalize(img_pillow)
    
    
    prob_list = []
    print('{}  predicting patches'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_equalizer.crop(coor_pil)
        tmp_image_patch_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])(tmp_image_patch)
        
        tmp_data = tmp_image_patch_tensor.type(torch.FloatTensor).unsqueeze(0)
        with torch.no_grad():
            tmp_output = net(tmp_data).squeeze(0).cpu().numpy()
            prob_list.append(tmp_output)
        print('{}  patch'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
            
    full_mask = merge_masks(prob_list, num_patch, img_pillow.size[0], img_pillow.size[1])
    
    if flag_mask_edge:
        full_mask[mask_edge] = 0
    full_mask_bin = full_mask > bin_thresh
    
    if np.sum(full_mask_bin) > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False
    print('{}  finished'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    return label, full_mask, RGB_equalizer


def predict_photo_streak(net, img_pillow, num_patch, patch_size, bin_thresh=0.2, thresh_num_pixel=1, flag_mask_edge=True):
    """ Predict if the pillow image have the defect under investigation
    Args:
    ------
      net: pytorch net with cpu device
    """
    print('{}  Start'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    #radius = 920
    radius = 910
    y_grid,x_grid = np.ogrid[-a_center:img_size_mask-a_center, -b_center:img_size_mask-b_center]
    mask_edge = x_grid*x_grid + y_grid*y_grid >= radius*radius
    
    pos = [(i,j) for i in range(num_patch) for j in range(num_patch)]
    RGB_equalizer = ImageOps.equalize(img_pillow)
    
    
    prob_list = []
    print('{}  predicting patches'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_equalizer.crop(coor_pil)
        tmp_image_patch_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])(tmp_image_patch)
        
        tmp_data = tmp_image_patch_tensor.type(torch.FloatTensor).unsqueeze(0)
        with torch.no_grad():
            tmp_output = net(tmp_data).squeeze(0).cpu().numpy()
            prob_list.append(tmp_output)
        print('{}  patch'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
            
    full_mask = merge_masks(prob_list, num_patch, img_pillow.size[0], img_pillow.size[1])
    
    if flag_mask_edge:
        full_mask[mask_edge] = 0
    full_mask_bin = full_mask > bin_thresh
    
    if np.sum(full_mask_bin) > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False
    print('{}  finished'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    return label, full_mask, RGB_equalizer


def predict_edge_arcing_generic(net, img_pillow, num_patch, patch_size, bin_thresh=0.2, thresh_num_pixel=1, flag_mask_edge=False, flag_mask_inner=True):
    """ Predict if the pillow image have the defect under investigation
    Args:
    ------
      net: pytorch net with cpu device
    """
#     print('{}  Start'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    #radius = 920
    radius_inner = 920
    radius_outer = 960
    
    y_grid,x_grid = np.ogrid[-a_center:img_size_mask-a_center, -b_center:img_size_mask-b_center]
    mask_ignore = x_grid*x_grid + y_grid*y_grid >= radius_outer*radius_outer
    if flag_mask_edge:
        mask_ignore = x_grid*x_grid + y_grid*y_grid >= radius_inner*radius_inner
    elif flag_mask_inner:
        mask_ignore = np.logical_or(mask_ignore, (x_grid*x_grid + y_grid*y_grid <= radius_inner*radius_inner))

    pos = [(i,j) for i in range(num_patch) for j in range(num_patch)]
    RGB_equalizer = ImageOps.equalize(img_pillow)
    
    
    prob_list = []
    for tmp_pos in pos:
        coor_pil = rowcolumn2coor(tmp_pos[0], tmp_pos[1], patch_size)
        tmp_image_patch = RGB_equalizer.crop(coor_pil)
        tmp_image_patch_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])(tmp_image_patch)
        
        tmp_data = tmp_image_patch_tensor.type(torch.FloatTensor).unsqueeze(0)
        with torch.no_grad():
            tmp_output = net(tmp_data).squeeze(0).cpu().numpy()
            prob_list.append(tmp_output)
            
    full_mask = merge_masks(prob_list, num_patch, img_pillow.size[0], img_pillow.size[1])
    
    full_mask[mask_ignore] = 0
    full_mask_bin = full_mask > bin_thresh
    
    if np.sum(full_mask_bin) > thresh_num_pixel:
        # Defect exists
        label = True
    else:
        label = False
#     print('{}  finished'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    return label, full_mask, RGB_equalizer


def predict_mask_loss(net, img_pillow, num_patch, patch_size, bin_thresh=0.2, thresh_num_pixel=1, 
                      flag_mask_edge=False, flag_mask_inner=False, flag_equalizer=False):
    """ Predict if the pillow image have the defect under investigation
    Args:
    ------
      net: pytorch net with cpu device
    """
#     print('{}  Start'.format(datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')))
    img_size_mask = 2048
    a_center, b_center = 1024, 1024
    #radius = 920
    radius_inner = 920
    radius_outer = 960
    
    y_grid,x_grid = np.ogrid[-a_center:img_size_mask-a_center, -b_center:img_size_mask-b_center]
    mask_ignore = x_grid*x_grid + y_grid*y_grid >= radius_outer*radius_outer
    if flag_mask_edge:
        mask_ignore = x_grid*x_grid + y_grid*y_grid >= radius_inner*radius_inner
    elif flag_mask_inner:
        mask_ignore = np.logical_or(mask_ignore, (x_grid*x_grid + y_grid*y_grid <= radius_inner*radius_inner))

    pos = [(i,j) for i in range(num_patch) for j in range(num_patch)]
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
            ])(tmp_image_patch)
        
        tmp_data = tmp_image_patch_tensor.type(torch.FloatTensor).unsqueeze(0)
        with torch.no_grad():
            tmp_output = net(tmp_data).squeeze(0).cpu().numpy()
            prob_list.append(tmp_output)
            
    full_mask = merge_masks(prob_list, num_patch, img_pillow.size[0], img_pillow.size[1])
    
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

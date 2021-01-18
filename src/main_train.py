import os
import sys
import numpy as np
import pandas as pd
import datetime
import time
import glob
import cv2
import torchvision.utils as vutils

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from src.unet.unet import UNet
from src.unet.train import train_nn
from src.unet.util_image import get_square, hwc_to_chw, normalize, merge_masks
from src.config import get_arguments

import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


from Utils.transform import *
from Utils.pillowhelper import *

from src.unet.train import split_train_val, train_nn_2
from src.unet.dataset import DatasetMaskLoss


def get_unet_model(opt):
    net = UNet(n_channels=opt.num_channel, n_classes=1)
    if torch.cuda.is_available():
        print('Move model to gpu')
        device = 'cuda'
    else:
        raise Exception('Please train on GPUs!')
        
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.cuda()
    return device, net


if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    
    # Additional parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda
    image_size = opt.height
    patch_size = int(opt.height / opt.num_patch)
    label_name_to_value = {'_background_': 0, opt.defect_name: 1}
    num_train = len(glob.glob(opt.input_json+'/*.json'))
    
    # Get unet model
    device, net = get_unet_model(opt)
    
    # Load training dataset
    ids = glob.glob(opt.input_json+'/*.json')
    ids = ((id, (i, j))  for id in ids for i in range(opt.num_patch) for j in range(opt.num_patch))
    iddataset = split_train_val(ids, num_val=0)
    dataset = DatasetMaskLoss(iddataset['train'], image_size, patch_size, label_name_to_value, flag_color_jitter=opt.flag_color_jitter)
    ds_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    print('#Training patches:', len(iddataset['train']))
    
    # Train unet
    loss_list, best_epoch, best_model, final_model = train_nn_2(net, device, ds_loader, opt.lr, opt.num_epochs)
    
    # Save the model
    if opt.save_dir == 'none':
        print('Model not saved!')
    else:
        dt_str = datetime.datetime.now().strftime('%Y%m%d-%H:%M')
        save_to = os.path.join(opt.save_dir, '{}-ds{}-epoch{}-lr{}-loss{}.pth'.format(
            dt_str, num_train, best_epoch, opt.lr, min(loss_list)
        ))
        try:
            torch.save(best_model, save_to)
            print('Weights have been saved to {}'.format(save_to))
        except:
            raise Exception('Please create the save folder!')
    

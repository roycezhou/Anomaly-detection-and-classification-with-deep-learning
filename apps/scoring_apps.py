import os
import sys
import shutil
import base64
import time
from collections import OrderedDict

import numpy as np
import torch
import cv2
from PIL import Image
sys.path.append(wd_dir)
from unet.unet import UNet
from unet.util_image import get_square, hwc_to_chw, normalize

class scoringApp:
    def __init__(self, image_base64,imagename):
        self.image_base64 = image_base64
        self.imagename = imagename

    def merge_masks(self,prob_list):
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

    def mask_to_image(self,mask):
        return Image.fromarray((mask * 255).astype(np.uint8))

    def imageprepare(self,text):
        output = None
        nparr = np.fromstring(base64.b64decode(text), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return RGB

    def EdgeArcingScoring(self):
        flag_check_B_channel = True
        #h_raw, w_raw = 2048, 2048
        num_patch = 4
        step = self.imagename.split('::')[1]
        level = self.imagename.split('::')[2]
        if level == '03':
            bin_thresh = 0.15
            num_pixel_thresh = 5
            brightness_thresh = 100
        else:
            bin_thresh = 0.1
            num_pixel_thresh = 5
        
        label_to_int = {'good':0, 'bad':1}
        overlay_color=[255, 0, 0]
        alpha = 0.6

        # modify model (trained on multiple gpu) and load weights to cpu
        net = UNet(n_channels=1, n_classes=1)
        state_dict = torch.load(model_dir, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.eval()
        pos = [(i,j) for i in range(num_patch) for j in range(num_patch)]

        imgtext = self.image_base64 + '==='
        imgRGB = self.imageprepare(imgtext)
        imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
        imgarr = np.expand_dims(np.dot(imgBGR[...,:3], [0.2989, 0.5870, 0.1140]), axis=2)
        imgarr = normalize(imgarr)
        imgs_patch = [(tmp_pos, hwc_to_chw(get_square(imgarr, tmp_pos))) for tmp_pos in pos]

        prob_list = []
        tic=time.time()
        for tmp_pos, patch in imgs_patch:
            tmp_data = torch.from_numpy(patch).type(torch.FloatTensor).unsqueeze(0)
            with torch.no_grad():
                tmp_output = net(tmp_data).squeeze(0).cpu().numpy()
                prob_list.append(tmp_output)
        toc=time.time()
        print('the mask finish time:' + str(toc - tic))
        full_mask = self.merge_masks(prob_list)

        # Modify the full mask so that the probability of non-edge region is zero
        a_center, b_center = 1024, 1024
        img_size_mask = 2048
        radius = 920
        y_grid,x_grid = np.ogrid[-a_center:img_size_mask-a_center, -b_center:img_size_mask-b_center]
        mask_non_edge = x_grid*x_grid + y_grid*y_grid <= radius*radius
        full_mask[mask_non_edge] = 0

        full_mask_bin = full_mask > bin_thresh
        num_positive_pixel = np.sum(full_mask_bin)
        flag_positive = (full_mask.max() > bin_thresh) and (num_positive_pixel > num_pixel_thresh)
        if (level == '03') and flag_positive:
            detected_pixels = imgRGB[full_mask_bin]
            brightness = 0.2126 * detected_pixels[:,0] + 0.7152 * detected_pixels[:,1] + 0.0722 * detected_pixels[:,2]
            if brightness.mean() > brightness_thresh:
                flag_positive = False
        positive = int(flag_positive)
        negative = 1 - positive

        output_mask = base64.b64encode(full_mask)

        #string_output = str(negative)+","+str(positive)
        return positive,output_mask

    def DummyAppScoring(self):
        return 'DummyApp output'

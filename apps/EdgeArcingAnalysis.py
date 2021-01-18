import os
import sys
import shutil
import base64
from collections import OrderedDict

import numpy as np
import torch
import cv2
from PIL import Image
sys.path.append(wd_dir)
from unet.unet import UNet
from unet.util_image import get_square, hwc_to_chw, normalize

def merge_masks(prob_list):
    new = np.zeros((h_raw, w_raw), np.float32)
    h_patch, w_patch = int(h_raw/num_patch), int(w_raw/num_patch)
    counter = 0
    for i in range(num_patch):
        for j in range(num_patch):
            new[i*h_patch:(i+1)*h_patch, j*w_patch:(j+1)*w_patch] = prob_list[counter]
            counter+=1
    return new
    
def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def imageprepare(text):
    output = None
    nparr = np.fromstring(base64.b64decode(text), np.uint8)
    img=cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return RGB

h_raw, w_raw = 2048, 2048
num_patch = 4
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

imgtext = sys.argv[1] + '==='
imgRGB = imageprepare(imgtext)
imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
imgarr = np.expand_dims(np.dot(imgBGR[...,:3], [0.2989, 0.5870, 0.1140]), axis=2)
imgarr = normalize(imgarr)
imgs_patch = [(tmp_pos, hwc_to_chw(get_square(imgarr, tmp_pos))) for tmp_pos in pos]

prob_list = []
for tmp_pos, patch in imgs_patch:
    tmp_data = torch.from_numpy(patch).type(torch.FloatTensor).unsqueeze(0)
    with torch.no_grad():
        tmp_output = net(tmp_data).squeeze(0).cpu().numpy()
        prob_list.append(tmp_output)
full_mask = merge_masks(prob_list)
full_mask_bin = full_mask > bin_thresh
num_positive_pixel = np.sum(full_mask_bin)

flag_positive = (full_mask.max() > bin_thresh) and (num_positive_pixel > num_pixel_thresh)
positive = int(flag_positive)
negative = 1 - positive

if flag_positive:
    filename = designid+':'+lotid+':'+waferid
    result_folder = detected_dir + filename
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
    os.mkdir(result_folder)

    result = mask_to_image(full_mask_bin)
    result.save(result_folder+'/mask.jpg')
    out = imgBGR.copy()
    img_layer = imgBGR.copy()
    img_layer[full_mask_bin] = overlay_color
    out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    cv2.imwrite(result_folder+'/overlay.jpg', out)
    cv2.imwrite(result_folder+'/origin.jpg', imgBGR.copy())

    with open(result_folder+'/result.txt', 'w+') as f:
        f.write(str(full_mask.max()) + str(num_positive_pixel))

string_output = str(negative)+","+str(positive)
print(string_output)
        

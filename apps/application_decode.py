import os
dir = os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], ".."))

import sys
import shutil
import base64
from collections import OrderedDict
import time
import pytz
from datetime import datetime
from pytz import timezone
import numpy as np
import torch
import torchvision.transforms as transforms
import io
from PIL import Image

sys.path.append(wd_dir)
from unet.unet import UNet
from unet.util_image import get_square, hwc_to_chw, normalize



if __name__ == "__main__":
    for line in sys.stdin:
        app_name = line[-4:]
        # app=str(app, encoding="utf-8")
        model_dir = model_dir_dic[app_name]
        num_channel = 3
        if(app_name == 'Edge'):
            num_channel = 1

        # initate the torch unet and loading the weight for different application
        net = UNet(n_channels=num_channel, n_classes=1)
        state_dict = torch.load(model_dir, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.eval()
        patchstring = base64.b64decode(line[:-4])
        if (app_name == 'Edge'):
            patch = np.frombuffer(patchstring, dtype=float).reshape((1, 512, 512))
            tmp_data = torch.from_numpy(patch).type(torch.FloatTensor).unsqueeze(0)

        else:
            patch = Image.open(io.BytesIO(patchstring))
            image_patch_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])(patch)

            tmp_data = image_patch_tensor.type(torch.FloatTensor).unsqueeze(0)

        with torch.no_grad():
            tmp_output = net(tmp_data).cpu().squeeze(0).numpy().astype(float)
            print(base64.b64encode(tmp_output.tostring()))
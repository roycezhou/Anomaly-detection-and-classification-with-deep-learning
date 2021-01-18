import numpy as np
import torch
import torch.nn.functional as F

from .dice_loss import dice_coeff

def eval_net(net, dataset):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = np.array(b[0]).astype(np.float32)
        true_mask = np.array(b[1]).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        img = img.cuda()
        true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = (mask_pred > 0.5).float()

        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / (i + 1)

import sys
import io
import base64
from PIL import Image, ImageOps
import json
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from .util_image import get_square, hwc_to_chw, normalize
from .util_labelme import shapes_to_label, img_b64_to_arr
from Utils.pillowhelper import rowcolumn2coor

shape_image = (2048, 2048)

def get_image_array_from_json(dir_json, num_channels):
    with open(dir_json, "r") as f:
        data = json.load(f)
    imageData = data['imageData']
    img = img_b64_to_arr(imageData)
    if num_channels == 1:
        img = np.expand_dims(np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]), axis=2)
    assert img.shape[:2] == shape_image
    return img


def get_pil_img_and_mask_from_json(dir_json, label_name_to_value):
    with open(dir_json, "r") as f:
        data = json.load(f)
    img_b64 = data['imageData']
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_pil = Image.open(f)
    
    mask_shapes = data['shapes']
    mask = shapes_to_label(img_pil.size, mask_shapes, label_name_to_value)
    return img_pil, mask


def get_image_mask_from_json(dir_json, label_name_to_value):
    """ Parse a json file to get both image and its corresponding mask (groundtruth). 
    Notes: 
    (1) In the json file, images have three channels and shape of image array is like (H,W,C);
    (2) Number of classes is limited to 256 since we are using uint8
    (3) The data type of mask should be float since it will be compared with prediction whose
        data type is float
    Arg:
    ------
    
    Return:
    -------
    img_array, mask
    """
    with open(dir_json, "r") as f:
        data = json.load(f)
    imageData = data['imageData']
    img_array = img_b64_to_arr(imageData)
    image_size = img_array.shape
    mask = shapes_to_label(image_size, data['shapes'], label_name_to_value)
    return img_array, mask


def get_mask_array_from_json(dir_json, label_name_to_value):
    with open(dir_json, "r") as f:
        data = json.load(f)
    mask = shapes_to_label(shape_image, data['shapes'], label_name_to_value)
    return mask

def cropped_imgs_to_patch(ids, num_channels):
    for id, pos in ids:
        im = get_image_array_from_json(id, num_channels)
        yield get_square(im, pos)
        
def cropped_mask_to_patch(ids, label_name_to_value):
    for id, pos in ids:
        im = get_mask_array_from_json(id, label_name_to_value)
        yield get_square(im, pos)

def dataset_generator(ids, label_name_to_value, num_channels):
    imgs = cropped_imgs_to_patch(ids, num_channels)
    imgs = map(hwc_to_chw, imgs)
    imgs = map(normalize, imgs)
    masks = cropped_mask_to_patch(ids, label_name_to_value)
    return zip(imgs, masks)


class DatasetDefocus(Dataset):
    def __init__(self, filename_with_patch, image_size, patch_size, label_name_to_value):
        self.fn_with_patch = filename_with_patch
        self.image_size = image_size
        self.patch_size = patch_size
        self.label_name_to_value = label_name_to_value
        
    def __len__(self):
        return len(self.fn_with_patch)
    
    def __getitem__(self, idx):
        fn_patch = self.fn_with_patch[idx]
        fn, patch_idx = fn_patch[0], fn_patch[1]
        image, mask = get_pil_img_and_mask_from_json(fn, self.label_name_to_value)
        image = ImageOps.equalize(image)
        coor_pil = rowcolumn2coor(patch_idx[0], patch_idx[1], self.patch_size)
        image_patch = image.crop(coor_pil)
        mask_patch = np.array(Image.fromarray(mask).crop(coor_pil)).astype(np.float32)
        image_patch_tensor = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])(image_patch)

        sample = {'image': image_patch_tensor, 'mask': mask_patch, 'filename': fn, 'rowcol': patch_idx}
        return sample
    
    
class DatasetComet(Dataset):
    def __init__(self, filename_with_patch, image_size, patch_size, label_name_to_value):
        self.fn_with_patch = filename_with_patch
        self.image_size = image_size
        self.patch_size = patch_size
        self.label_name_to_value = label_name_to_value
        
    def __len__(self):
        return len(self.fn_with_patch)
    
    def __getitem__(self, idx):
        fn_patch = self.fn_with_patch[idx]
        fn, patch_idx = fn_patch[0], fn_patch[1]
        image, mask = get_pil_img_and_mask_from_json(fn, self.label_name_to_value)
        # image = ImageOps.equalize(image)
        coor_pil = rowcolumn2coor(patch_idx[0], patch_idx[1], self.patch_size)
        image_patch = image.crop(coor_pil)
        mask_patch = np.array(Image.fromarray(mask).crop(coor_pil)).astype(np.float32)
        image_patch_tensor = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])(image_patch)

        sample = {'image': image_patch_tensor, 'mask': mask_patch, 'filename': fn, 'rowcol': patch_idx}
        return sample
    
    
class DatasetSmudge(Dataset):
    def __init__(self, filename_with_patch, image_size, patch_size, label_name_to_value):
        self.fn_with_patch = filename_with_patch
        self.image_size = image_size
        self.patch_size = patch_size
        self.label_name_to_value = label_name_to_value
        
    def __len__(self):
        return len(self.fn_with_patch)
    
    def __getitem__(self, idx):
        fn_patch = self.fn_with_patch[idx]
        fn, patch_idx = fn_patch[0], fn_patch[1]
        image, mask = get_pil_img_and_mask_from_json(fn, self.label_name_to_value)
        # image = ImageOps.equalize(image)
        coor_pil = rowcolumn2coor(patch_idx[0], patch_idx[1], self.patch_size)
        image_patch = image.crop(coor_pil)
        mask_patch = np.array(Image.fromarray(mask).crop(coor_pil)).astype(np.float32)
        image_patch_tensor = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])(image_patch)

        sample = {'image': image_patch_tensor, 'mask': mask_patch, 'filename': fn, 'rowcol': patch_idx}
        return sample
    
    
class DatasetCactus(Dataset):
    def __init__(self, filename_with_patch, image_size, patch_size, label_name_to_value):
        self.fn_with_patch = filename_with_patch
        self.image_size = image_size
        self.patch_size = patch_size
        self.label_name_to_value = label_name_to_value
        
    def __len__(self):
        return len(self.fn_with_patch)
    
    def __getitem__(self, idx):
        fn_patch = self.fn_with_patch[idx]
        fn, patch_idx = fn_patch[0], fn_patch[1]
        image, mask = get_pil_img_and_mask_from_json(fn, self.label_name_to_value)
        # image = ImageOps.equalize(image)
        coor_pil = rowcolumn2coor(patch_idx[0], patch_idx[1], self.patch_size)
        image_patch = image.crop(coor_pil)
        mask_patch = np.array(Image.fromarray(mask).crop(coor_pil)).astype(np.float32)
        image_patch_tensor = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])(image_patch)

        sample = {'image': image_patch_tensor, 'mask': mask_patch, 'filename': fn, 'rowcol': patch_idx}
        return sample
    
    
class DatasetStreak(Dataset):
    def __init__(self, filename_with_patch, image_size, patch_size, label_name_to_value):
        self.fn_with_patch = filename_with_patch
        self.image_size = image_size
        self.patch_size = patch_size
        self.label_name_to_value = label_name_to_value
        
    def __len__(self):
        return len(self.fn_with_patch)
    
    def __getitem__(self, idx):
        fn_patch = self.fn_with_patch[idx]
        fn, patch_idx = fn_patch[0], fn_patch[1]
        image, mask = get_pil_img_and_mask_from_json(fn, self.label_name_to_value)
        # image = ImageOps.equalize(image)
        coor_pil = rowcolumn2coor(patch_idx[0], patch_idx[1], self.patch_size)
        image_patch = image.crop(coor_pil)
        mask_patch = np.array(Image.fromarray(mask).crop(coor_pil)).astype(np.float32)
        image_patch_tensor = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])(image_patch)

        sample = {'image': image_patch_tensor, 'mask': mask_patch, 'filename': fn, 'rowcol': patch_idx}
        return sample
    
    
class DatasetEdgeArcing(Dataset):
    def __init__(self, filename_with_patch, image_size, patch_size, label_name_to_value):
        self.fn_with_patch = filename_with_patch
        self.image_size = image_size
        self.patch_size = patch_size
        self.label_name_to_value = label_name_to_value
        
    def __len__(self):
        return len(self.fn_with_patch)
    
    def __getitem__(self, idx):
        fn_patch = self.fn_with_patch[idx]
        fn, patch_idx = fn_patch[0], fn_patch[1]
        image, mask = get_pil_img_and_mask_from_json(fn, self.label_name_to_value)
        # image = ImageOps.equalize(image)
        coor_pil = rowcolumn2coor(patch_idx[0], patch_idx[1], self.patch_size)
        image_patch = image.crop(coor_pil)
        mask_patch = np.array(Image.fromarray(mask).crop(coor_pil)).astype(np.float32)
        image_patch_tensor = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])(image_patch)

        sample = {'image': image_patch_tensor, 'mask': mask_patch, 'filename': fn, 'rowcol': patch_idx}
        return sample
    
class DatasetMaskLoss(Dataset):
    def __init__(self, filename_with_patch, image_size, patch_size, label_name_to_value, 
                 flag_color_jitter=False, flag_equalizer=False):
        self.fn_with_patch = filename_with_patch
        self.image_size = image_size
        self.patch_size = patch_size
        self.label_name_to_value = label_name_to_value
        self.flag_color_jitter = flag_color_jitter
        self.flag_equalizer = flag_equalizer
        
    def __len__(self):
        return len(self.fn_with_patch)
    
    def __getitem__(self, idx):
        fn_patch = self.fn_with_patch[idx]
        fn, patch_idx = fn_patch[0], fn_patch[1]
        image, mask = get_pil_img_and_mask_from_json(fn, self.label_name_to_value)
        if self.flag_equalizer:
            image = ImageOps.equalize(image)
        coor_pil = rowcolumn2coor(patch_idx[0], patch_idx[1], self.patch_size)
        image_patch = image.crop(coor_pil)
        mask_patch = np.array(Image.fromarray(mask).crop(coor_pil)).astype(np.float32)
        if self.flag_color_jitter:
            image_patch_tensor = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                ])(image_patch)
        else:
            image_patch_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                ])(image_patch)            

        sample = {'image': image_patch_tensor, 'mask': mask_patch, 'filename': fn, 'rowcol': patch_idx}
        return sample
    
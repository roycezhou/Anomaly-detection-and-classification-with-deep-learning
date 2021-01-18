import glob
from torch.utils.data import Dataset
from .util_image import hwc_to_chw
from .dataset import get_image_mask_from_json

class ScratchDataset(Dataset):
    """Scratch Dataset"""
    def __init__(self, root_dir, label_name_to_value, num_channels=3, transform=None):
        """
        Arg:
          root_dir: root directory of labelme json files
        """
        self.json_lblme = glob.glob(root_dir + '/*.json')
        self.root_dir = root_dir
        self.num_channels = num_channels
        self.label_name_to_value = label_name_to_value
        self.transform = transform
        
    def __len__(self):
        return len(self.json_lblme)
    
    def __getitem__(self, idx):
        json_filename = self.json_lblme[idx]
        image, mask = get_image_mask_from_json(json_filename, self.label_name_to_value)
        image = hwc_to_chw(image) / 255
        sample = {'image': image, 'filename': os.path.basename(json_filename), 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)
        return sample
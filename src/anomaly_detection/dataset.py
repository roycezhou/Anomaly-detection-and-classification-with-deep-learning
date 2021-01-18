import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from preprocessing import Equalizer

transform_default = transforms.Compose([
            transforms.Resize((256, 256)),
            Equalizer(),
            transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
        ])


class DatasetFromDir(Dataset):
    def __init__(self, root_dir, size=None, transform=None):
        self.root_dir = root_dir
        filename_all = glob.glob(self.root_dir+'/**/*.jpg')
        if size:
            size = min(size, len(filename_all))
            self.filename_all = filename_all[:size]
        else:
            self.filename_all = filename_all
        
        if transform:
            self.transform = transform
        else:
            self.transform = transform_default
        
    
    def __len__(self):
        return len(self.filename_all)
    
    def __getitem__(self, idx):
        filename = os.path.basename(self.filename_all[idx])
        image = Image.open(self.filename_all[idx])
        image = self.transform(image)
        sample = {'image': image, 'filename': filename}
        return sample
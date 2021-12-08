from torch.utils.data import Dataset
import glob
from PIL import Image
import os

class TrainSet(Dataset):
    def __init__(self, path = '.', transform = None):
        self.path = path
        self.data = [img_path for img_path in sorted(glob.glob(os.path.join(self.path, '*.png')))]
        self.transform = transform
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        img = Image.open(self.data[i])
        img = self.transform(img)
        return img, int(self.data[i].split('/')[-1].split('_')[0])
    
class TestSet(Dataset):
    def __init__(self, path = '.', transform = None):
        self.path = path
        self.data = [img_path for img_path in sorted((glob.glob(os.path.join(self.path, '*.png'))))]
        self.transform = transform
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        img = Image.open(self.data[i])
        img = self.transform(img)
        return img, self.data[i].split('/')[-1]

if __name__ == '__main__':
    from constant import *
    ts = TrainSet(train_dir)
    tts = TestSet(valid_dir)
    print(len(ts), len(tts))

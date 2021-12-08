import glob
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

def mask_target(path):
    mask = Image.open(path)
    mask = np.array(mask).astype(np.uint8)
    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    masks = np.zeros((512,512))
    masks[mask == 3] = 0  # (Cyan: 011) Urban land 
    masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
    masks[mask == 5] = 2  # (Purple: 101) Rangeland 
    masks[mask == 2] = 3  # (Green: 010) Forest land 
    masks[mask == 1] = 4  # (Blue: 001) Water 
    masks[mask == 7] = 5  # (White: 111) Barren land 
    masks[mask == 0] = 6  # (Black: 000) Unknown
    masks[mask == 4] = 6  # (Red: 100) Unknown
    
    return torch.LongTensor(masks)

class TrainSet(Dataset):
    def __init__(self, path, transform=None, phase = 'train'):
        self.length = 2000 if phase == 'train' else 257
        self.filenames = [(os.path.join(path, "{:04}_sat.jpg".format(i)), os.path.join(path, "{:04}_mask.png".format(i))) for i in range(self.length)]
        self.transform = transform
        if phase == 'train':
            self.counts = self.__compute_class_probability()
    
    def __compute_class_probability(self):
        counts = [0 for i in range(7)]
        plus = [3, 6, 5, 2, 1, 7, 0, 4]
        for _, mask_name in self.filenames:
            mask = Image.open(mask_name)
            mask = np.array(mask).astype(np.uint8)
            mask = (mask >= 128).astype(int)
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
            cnt = np.bincount(mask.flatten())
            cnt = np.pad(cnt, (0, 8 - len(cnt)), mode='constant')
            for i in range(7):
                counts[i] += cnt[plus[i]]
            counts[6] += cnt[4]
        return counts

    def get_class_probability(self):
        values = np.array(self.counts)
        p_values = values / np.sum(values)

        return torch.Tensor(p_values)

    def __getitem__(self, idx):
        sat_name, mask_name= self.filenames[idx]
        sat = Image.open(sat_name).convert('RGB')
        if self.transform is not None:
            sat = self.transform(sat)

        return sat, mask_target(mask_name)

    def __len__(self):
        return self.length


class TestSet(Dataset):
    def __init__(self, path, transform = None):
        self.filenames = sorted(glob.glob(os.path.join(path, '*.jpg')))
        self.transform = transform
                              
    def __getitem__(self, idx):
        sat_name = self.filenames[idx]
        sat = Image.open(sat_name).convert('RGB')
        if self.transform is not None:
            sat = self.transform(sat)
        return sat, sat_name.split('/')[-1].split('_')[0]

    def __len__(self):
        return len(self.filenames)

if __name__ == '__main__':
    from constant import *
    train_set = TrainSet(path = train_dir, transform=transform)
    valid_set = TrainSet(path = valid_dir, transform=transform, phase = 'val')

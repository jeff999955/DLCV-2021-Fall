from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import pandas as pd
from constant import *


class OfficeHome(Dataset):
    def __init__(self, path=office_train_dir, csv_path=office_train_dir + '.csv', transform=tfm):
        self.path = path
        self.data = pd.read_csv(csv_path, index_col="id")
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = Image.open(os.path.join(
            self.path, self.data.loc[i].filename)).convert('RGB')
        img = self.transform(img)
        label = -1 
        if self.data.loc[i].label in office_labels:
            label = office_labels[self.data.loc[i].label]
        return img, label, self.data.loc[i].filename

class Mini_ImageNet(Dataset):
    def __init__(self, path=mini_train_dir, csv_path=mini_train_dir + '.csv', transform=tfm, mode = "train"):
        self.path = path
        self.data = pd.read_csv(csv_path, index_col="id")
        self.transform = transform
        if mode == "train":
            self.mapping = mini_train_labels
            self.listed = mini_train_list_labels
        else:
            self.mapping = mini_val_labels
            self.listed = mini_val_list_labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = Image.open(os.path.join(
            self.path, self.data.loc[i].filename)).convert('RGB')
        img = self.transform(img)
        return img, self.mapping[self.data.loc[i].label], self.data.loc[i].filename


if __name__ == "__main__":
    oh = OfficeHome()
    img, label, filename = oh[0]
    print(img.shape, label, filename)

    mi = Mini_ImageNet()
    img, label, filename = mi[0]
    print(img.shape, label, filename)

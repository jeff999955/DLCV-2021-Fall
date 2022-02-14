import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class CTDataset(Dataset):
    def __init__(self, json_path, data_path, mode = "train"):
        with open(os.path.join(data_path, 'records_train.json')) as f:
            json_data = json.load(f)["datainfo"]
        self.root = os.path.join(data_path, mode)
        self.keys = list(json_data.keys())
        self.json_data = json_data

    def __getitem__(self, idx):
        """
        Args:
            idx (int): index for access

        Returns:
            key (str): idx-th key
            data (ndarray): the image with dimension (3, 512, 512)
            label (int): the label {-1, 0, 1}
            coords (str): the coordinates encoded in x_i,y_i/x_{i+1},y_{i+1} form
        """
        key = self.keys[idx]
        loc = self.json_data[key] 
        td = np.load(os.path.join(self.root, loc["path"]))
        data = np.stack([td, td, td])
        label = loc["label"]
        coords = "/".join([f"{i[0]},{i[1]}" for i in (loc["coords"])])
        return key, data, label, coords

    def __len__(self):
        return len(self.keys)

if __name__ == "__main__":
    ds = CTDataset('./skull/records_train.json', './skull')
    print(ds[0])
    cnt = 0
    dl = DataLoader(ds, batch_size = 8, shuffle = False, num_workers = 8)
    device = "cuda"
    for batch in dl:
        key, data, target, coords = batch
        try:
            if coords[0] != "":
                print(data.shape)
                print(coords[0])
                break
        except Exception as e:
            print(len(key))
            print(len(data))
            print(len(target))
            print(coords)
            print(e)
            break

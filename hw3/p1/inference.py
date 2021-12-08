from dataset import TestSet
from torch.utils.data import DataLoader
import torch
from constant import *
import argparse
import numpy as np
import glob
import os
import timm
from pytorch_pretrained_vit import ViT
import torch.nn as nn
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class DV(nn.Module):
    def __init__(self):
        super(DV, self).__init__()
        model_name = 'B_16_imagenet1k'
        self.model = ViT(model_name)
        self.head = nn.Linear(1000, 37)
    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x


class V(nn.Module):
    def __init__(self):
        super(V, self).__init__()
        model_name = 'B_16_imagenet1k'
        self.model = ViT(model_name)
        self.model.fc = nn.Linear(768, 37)
    def forward(self, x):
        x = self.model(x)
        return x

class DT(nn.Module):
    def __init__(self):
        super(DT, self).__init__()
        self.model = timm.create_model('vit_base_patch16_384')
        self.head = nn.Linear(1000, 37)
    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x


def inference(args):
    test_set = TestSet(path = args.test_dir, transform = transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_name, raw_prediction, output = [], [], []
    for ckpt, m in [(args.T_path, None), (args.DT_path, DT), (args.V_path, V), (args.DV_path, DV)]:
        try:
            if m:
                model = m()
            else:
                model = timm.create_model('vit_base_patch16_384')
                model.head = torch.nn.Linear(768, 37)
            model.load_state_dict(torch.load(ckpt, map_location = 'cpu'))
            model = model.to(device)
            model.eval()
            name = []
            prediction = []
            print('inferencing on' + ckpt)
            for batch in test_loader:
                imgs, labels = batch
                with torch.no_grad():
                    logits = model(imgs.to(device))
                name.extend(labels)
                prediction.extend(logits.argmax(dim = -1).int().cpu().numpy())
            output_name = name
            raw_prediction.append(prediction)
        except Exception as e:
            print(e)
    
    raw_prediction = np.array(raw_prediction, np.int32)

    for sel in raw_prediction.T:
        output.append(np.bincount(sel).argmax())
    with open(args.out_csv, 'w') as f:
        f.write('filename, label\n')
        for i in range(len(output_name)):
            f.write('{},{}\n'.format(output_name[i], output[i]))
    try:
        parsed = [int(i.split('_')[0]) for i in output_name]
        n = 0
        for i in range(len(output)):
            n += 1 if parsed[i] == output[i] else 0
        return (n / len(output))
    except:
        return -1

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--test_dir', type=str,
                        default='/tmp2/b08902134/hw1_data/p1_data/testdata/val_50/', help='input files path')
    parser.add_argument('--out_csv', type=str, default='./test_pred.csv',
                        help='output csv file path')
    parser.add_argument('--T_path', type=str, default = './T.ckpt')
    parser.add_argument('--DT_path', type=str, default = './DT.ckpt')
    parser.add_argument('--V_path', type=str, default = './V.ckpt')
    parser.add_argument('--DV_path', type=str, default = './DV.ckpt')
    args = parser.parse_args()
    return args

def main():
    set_seed(1126)
    args = parse_args()
    accuracy = inference(args)
    if accuracy > 0:
        print("accuracy = ", accuracy)
    else:
        print("cannot calculate")

if __name__ == '__main__':
    main()

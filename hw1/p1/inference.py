from dataset import TestSet
from torch.utils.data import DataLoader
import torch
from constant import *
import argparse
from resnet101 import Classifier
from utils import get_device
import numpy as np
import glob
import os


def inference(args, checkpoint_path):
    test_set = TestSet(path = args.test_dir, transform = test_tfm)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    
    device = get_device()

    output_name, raw_prediction, output = [], [], []
    for ckpt in glob.glob(os.path.join(args.ckpt_dir,'resnet*.ckpt')):
        model = Classifier()
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
    
    raw_prediction = np.array(raw_prediction, np.int32)

    for sel in raw_prediction.T:
        output.append(np.bincount(sel).argmax())
    with open(args.out_csv, 'w') as f:
        f.write('image_id, label\n')
        for i in range(len(output_name)):
            f.write('{}, {}\n'.format(output_name[i], output[i]))

    parsed = [int(i.split('_')[0]) for i in output_name]
    n = 0
    for i in range(len(output)):
        n += 1 if parsed[i] == output[i] else 0
    return (n / len(output))

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--test_dir', type=str,
                        default='/tmp2/b08902134/hw1_data/p1_data/testdata/val_50/', help='input files path')
    parser.add_argument('--out_csv', type=str, default='./test_pred.csv',
                        help='output csv file path')
    parser.add_argument('--ckpt_dir', type=str, default='../', help='checkpoint files path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    accuracy = inference(args, checkpoint_path = '/tmp2/b08902134/hw1_sackpt/resnet101_5.ckpt')
    print("accuracy = ", accuracy)

if __name__ == '__main__':
    main()

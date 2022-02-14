import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import random
import argparse

from dataset import *
from constant import *
from model import Classifier


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def inference(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_set = OfficeHome(path=args.test_dir, csv_path=args.test_csv)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    output_name, raw_prediction, output = [], [], []
    labels = []
    for ckpt in glob.glob(os.path.join(args.ckpt_dir, '*.ckpt')):
        print(f"inferencing on {ckpt}")
        model = Classifier(n_classes=len(office_labels), dropout=True)
        model.load_state_dict(torch.load(ckpt, map_location = 'cpu'))
        model = model.to(device)
        model.eval()

        name, prediction = [], []
        lbls = []
        valid_accs = []
        for batch in tqdm(test_loader):
            imgs, l, filename = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
            acc = (logits.argmax(dim=-1) == l.to(device)).float().mean()
            name.extend(filename)
            prediction.extend(logits.argmax(dim=-1).int().cpu().numpy())
            lbls.extend(l.int().numpy())
            valid_accs.append(acc)
        valid_acc = sum(valid_accs) / len(valid_accs)
        print("individual accuracy on", ckpt, ":", valid_acc.item())
        output_name = name
        raw_prediction.append(prediction)
        labels = lbls

    raw_prediction = np.array(raw_prediction, np.int32)
    for sel in raw_prediction.T:
        output.append(np.bincount(sel).argmax())
    with open(args.out_csv, 'w') as f:
        f.write('id,filename,label\n')
        for i in range(len(output_name)):
            f.write('{},{},{}\n'.format(
                i,output_name[i], office_list_labels[output[i]]))
    try:
        cnt = 0
        for i in range(len(labels)):
            cnt += 1 if labels[i] == output[i] else 0
    except Exception as e:
        return -1
    return (cnt / len(labels))


def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--test_dir', type=str,
                        default='../hw4_data/office/val', help='input files path')
    parser.add_argument('--test_csv', type=str,
                        default='../hw4_data/office/val.csv', help='input files path')
    parser.add_argument('--out_csv', type=str, default='./test_pred.csv',
                        help='output csv file path')
    parser.add_argument('--ckpt_dir', type=str,
                        default='./C_kpt', help='checkpoint files path')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    return args


def main():
    set_seed(1126)
    args = parse_args()
    accuracy = inference(args)
    print("accuracy = ", accuracy)


if __name__ == '__main__':
    main()

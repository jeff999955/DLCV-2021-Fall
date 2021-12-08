import os
import argparse
from dataset import TestSet
import torch
from torch.utils.data import DataLoader
from model import DeepLabv3_ResNet101
from PIL import Image
from utils import *
from constant import *


color = [[0, 255, 255],[255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]

def main(args):
    device = get_device()
    model = DeepLabv3_ResNet101().to(device)
    model.load_state_dict(torch.load(args.ckpt))
    test_set = TestSet(args.img_dir, transform = transform)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    os.makedirs(args.save_dir, exist_ok=True)

    model.eval()
    for batch in test_loader:
        img, filename = batch
        img = img.to(device)
        with torch.no_grad():
            output = model(img)
        pred = output.argmax(dim = 1)
        for n in range(len(filename)):
            cur = pred[n].cpu().numpy()
            opimg = np.zeros((cur.shape[0], cur.shape[1], 3), dtype = np.uint8)
            for i in range(cur.shape[0]):
                for j in range(cur.shape[1]):
                    opimg[i,j] = color[cur[i][j]]
            im = Image.fromarray(opimg)
            im.save(os.path.join(args.save_dir, '{}.png'.format(filename[n])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default=valid_dir)
    parser.add_argument('--save_dir', type=str, default='output/')
    parser.add_argument('--ckpt', type=str, default='/tmp2/b08902134/hw1_checkpoint/DeepLabv3_ResNet101_20.ckpt')
    args = parser.parse_args()
    print(args)
    main(args)

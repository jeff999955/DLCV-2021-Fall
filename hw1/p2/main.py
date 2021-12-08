from train import train
from dataset import TrainSet
from torch.utils.data import DataLoader
from utils import same_seeds, get_device
import torch
from constant import *
from model import FCN32s, SegNet, DeepLabv3_ResNet101
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, ReduceLROnPlateau

def main():
    same_seeds(1126)

    batch_size = 8
    train_set = TrainSet(path = train_dir, transform=transform)
    valid_set = TrainSet(path = valid_dir, transform=transform, phase = 'val')
    train_loader= DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory = True)
    valid_loader= DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory = True)

    model = DeepLabv3_ResNet101()

    # weakly supervised
    # idea from https://github.com/say4n/pytorch-segnet
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight = 1.0 / train_set.get_class_probability())
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 1e-6, amsgrad = True)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 1e-6)

    # idea from 
    # https://medium.com/@hanrelan/a-non-experts-guide-to-image-segmentation-using-deep-neural-nets-dda5022f6282
    scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.9)
    # scheduler = LambdaLR(optimizer, lr_lambda = lambda i: (1 - i / 100) ** 0.9)
    best_mean_iou, best_epoch = train(model, get_device(), train_loader, valid_loader, criterion, optimizer, scheduler = None, n_epochs = 100)
    print(best_mean_iou, best_epoch)
    
if __name__ == '__main__':
    main()

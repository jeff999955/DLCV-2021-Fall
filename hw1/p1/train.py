import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from resnet101 import Classifier
from utils import same_seeds, get_device
from dataset import TrainSet
from constant import *

def train(model, device, train_loader, valid_loader, criterion, optimizer, scheduler = None, n_epochs = 15):
    model.to(device)
    best_acc = 0.0
    result = []
    for epoch in range(n_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
        train_loss = []
        train_accs = []
        model.train()
        for batch in train_loader:
            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)
            optimizer.step()
            if scheduler is not None and 8 < epoch < 11: scheduler.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print("[ Train | {:03d}/{:03d} ] loss = {:.5f}, acc = {:.5f}".format(epoch + 1, n_epochs, train_loss, train_acc))

        model.eval()
        valid_loss = []
        valid_accs = []

        for batch in valid_loader:
            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        print("[ Valid | {:03d}/{:03d} ] loss = {:.5f}, acc = {:.5f}".format(epoch + 1, n_epochs, valid_loss, valid_acc))
        result.append((epoch + 1, valid_loss, valid_acc))
        if valid_acc > best_acc:
            best_acc = valid_acc
        torch.save(model.state_dict(), '{}{}_{}.ckpt'.format(chkpt_dir, model.name, epoch + 1), _use_new_zipfile_serialization = False)
    return result

def main():
    train_set = TrainSet(path = train_dir, transform = train_tfm)
    valid_set = TrainSet(path = valid_dir, transform = test_tfm)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    device = get_device()
    model = Classifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    result = train(model, device, train_loader, valid_loader, criterion, optimizer, scheduler)
    for r in result:
        print(r)

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import transformers
import numpy as np
import random

from dataset import *
from constant import *
from model import Classifier

from byol import BYOL

project_name = "SSL-BYOL"
DEBUG = False
starting_epoch = 0
n_epochs = 1000
batch_size = 128
chkpt_dir = './ckpt'
os.makedirs(chkpt_dir, exist_ok = True)
os.makedirs(os.path.join(chkpt_dir, project_name), exist_ok = True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(1126) 


train_set = Mini_ImageNet()
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

if not torch.cuda.is_available():
    exit(1)
device = 'cuda'

model = Classifier()
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-3)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

learner = BYOL(model, 128, hidden_layer = 'model.avgpool')


for epoch in range(starting_epoch, n_epochs):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch + 1, n_epochs))
    train_loss = []
    model.train()
    for batch in tqdm(train_loader):
        imgs, _, _ = batch
        loss = learner(imgs.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        learner.update_moving_average()

        train_loss.append(loss.item())
    train_loss = sum(train_loss) / len(train_loss)
    scheduler.step(train_loss)

    print("[ Train | {:03d}/{:03d} ] ssl_loss = {:.5f}".format(epoch + 1, n_epochs, train_loss))
    torch.save({ 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, os.path.join(chkpt_dir, project_name, f"{epoch}.ckpt"))

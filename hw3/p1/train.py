from constant import *
from dataset import TrainSet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_pretrained_vit import ViT
from tqdm import tqdm
import os
import transformers
import timm
import numpy as np
import random
from conformer import ViC

DEBUG = False
n_epochs = 100
chkpt_dir = './ckpt'
os.makedirs(chkpt_dir, exist_ok = True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(1126) 


train_set = TrainSet(path = train_dir, transform = transform)
valid_set = TrainSet(path = valid_dir, transform = transform)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False, num_workers=8)

if not torch.cuda.is_available():
    exit(1)
device = 'cuda'


model = ViC()
model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing = 0.3)
optimizer = transformers.AdamW(model.parameters(), lr = 2e-5)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 500, num_training_steps = 6500)

best_acc = 0.0
result = []
for epoch in range(n_epochs):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch + 1, n_epochs))
    train_loss = []
    train_accs = []
    model.train()
    for batch in tqdm(train_loader):
        imgs, labels = batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
        optimizer.step()
        scheduler.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    print("[ Train | {:03d}/{:03d} ] loss = {:.5f}, acc = {:.5f}".format(epoch + 1, n_epochs, train_loss, train_acc))

    model.eval()
    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):
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
    torch.save(model.state_dict(), os.path.join(chkpt_dir, "Conformer", f'{epoch + 1}.ckpt'))


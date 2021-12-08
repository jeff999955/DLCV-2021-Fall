import random
import os
import sys
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import argparse

model_dir = './model'
data_path = '../hw2_data/digits'
os.makedirs(model_dir, exist_ok = True)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class CNNModel(nn.Module):
    def __init__(self, code_size=512, n_class=10):
        super(CNNModel, self).__init__()
        
        self.feature_extractor_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.feature_extractor_fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, code_size),
            nn.BatchNorm1d(code_size),
            nn.Dropout(),
            nn.ReLU(True)
        )
        
        self.class_classifier = nn.Sequential(
            nn.Linear(code_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, n_class),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(code_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def encode(self, x):
        feature = self.feature_extractor_conv(x)
        feature = feature.view(-1, 50 * 4 * 4)
        feature = self.feature_extractor_fc(feature)

        return feature

    def forward(self, x, alpha=1.0):
        feature = self.feature_extractor_conv(x)
        feature = feature.view(-1, 50 * 4 * 4)
        feature = self.feature_extractor_fc(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        
        return class_output, domain_output


class TrainSet(Dataset):
    def __init__(self, name, mode, transform):
        self.path = os.path.join(data_path, name, mode)
        self.data = sorted(os.listdir(self.path))
        df = pd.read_csv(os.path.join(data_path, name, mode + '.csv'))
        self.label = dict(zip(df.image_name, df.label))
        self.transform = transform
        
    def __getitem__(self, index):
        return self.transform(Image.open(os.path.join(self.path, self.data[index])).convert('RGB')), int(self.label[self.data[index]])

    def __len__(self):
        return len(self.data)

cuda = True
lr = 1e-3
batch_size = 128
image_size = 28
n_epoch = 50

same_seeds(1126)

src_tfm = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

target_tfm = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

parser = argparse.ArgumentParser()
parser.add_argument("--source", default = "svhn", type = str)
parser.add_argument("--target", default = "mnistm", type = str)
parser.add_argument("--mode", default = "source", type = str)
parser.add_argument("--n_iters", default = 100000, type = int)
args = parser.parse_args()

src_dataset_name = args.source
target_dataset_name = args.target
if args.mode == "target":
    src_dataset = TrainSet(target_dataset_name, 'train', src_tfm)
else:
    src_dataset = TrainSet(src_dataset_name, 'train', src_tfm)

src_loader = DataLoader(src_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
target_dataset = TrainSet(target_dataset_name, 'train', target_tfm)
target_loader = DataLoader(target_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)

print(src_dataset_name, src_dataset[0][0].shape)
print(target_dataset_name, target_dataset[0][0].shape)
train_target = args.mode == "adaptation"
print(f'{train_target}')

model = CNNModel()
optimizer = optim.Adam(model.parameters(), lr = lr)

if cuda:
    model = model.cuda()

for p in model.parameters():
    p.requires_grad = True

def test(dataset_name, model):
    target_dataset = TrainSet(target_dataset_name, 'test', target_tfm)
    dataloader = DataLoader(target_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)

    model.eval()

    if cuda:
        model = model.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, _ = model(t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu

from tqdm import tqdm

criterion = nn.NLLLoss()
device = 'cuda' if cuda else 'cpu'

best_acc = 0
best_loss = 1e15
iteration = 0


while iteration < args.n_iters:
    model.train()  
    optimizer.zero_grad()

    try:
        data, label = next(src_data_iter)
    except:
        src_data_iter = iter(src_loader)
        data, label = next(src_data_iter)

    data, label = data.to(device), label.to(device)
    src_batch_size = data.size(0)

    p = float(iteration) / (args.n_iters)
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    src_domain = torch.zeros((src_batch_size,), dtype=torch.long, device=device)

    class_output, domain_output = model(data, alpha)

    src_c_loss = criterion(class_output, label)
    src_d_loss = criterion(domain_output, src_domain)

    loss = src_c_loss + src_d_loss

    if not train_target:
        tgt_d_loss = torch.zeros(1)
    else:
        try:
            tgt_data, _ = next(tgt_data_iter)
        except:
            tgt_data_iter = iter(target_loader)
            tgt_data, _ = next(tgt_data_iter)

        tgt_data = tgt_data.to(device)
        tgt_batch_size = tgt_data.size(0)
        tgt_domain = torch.ones((tgt_batch_size,), dtype=torch.long, device=device)

        _, domain_output = model(tgt_data, alpha)
        tgt_d_loss = criterion(domain_output, tgt_domain)

        loss += tgt_d_loss

    loss.backward()
    optimizer.step()

    # Output training stats
    if (iteration+1) % 100 == 0:
        print('Iteration: {:5d} loss: {:.6f} loss_src_class: {:.6f} loss_src_domain: {:.6f} loss_tgt_domain: {:.6f}'.format(
            iteration + 1, 
            loss.item(), src_c_loss.item(), src_d_loss.item(), tgt_d_loss.item()))
        acc = test(target_dataset_name, model)
        print('Accuracy =', acc)

    # Save model checkpoints
    if (iteration+1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, '{}_{}_{}_{}.pth'.format(src_dataset_name, target_dataset_name, args.mode, iteration + 1)))

    iteration += 1

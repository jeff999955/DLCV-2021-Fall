import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from utils import *
from dataset import *
from model import *


def parse_args():
    parser = argparse.ArgumentParser()

    # training configuration.
    parser.add_argument('--episodes', default=600, type=int)
    parser.add_argument('--N_way', default=5, type=int, help='N_way (default: 5) for training')
    parser.add_argument('--N_shot', default=1, type=int, help='N_shot (default: 1) for training')
    parser.add_argument('--N_query', default=15, type=int, help='N_query (default: 15) for training')
    parser.add_argument('--matching_fn', default='parametric', type=str, choices=['euclidean', 'cosine', 'parametric'], help='distance matching function')

    parser.add_argument('--train_csv', type=str, default='../hw4_data/mini/train.csv', help="Training images csv file")
    parser.add_argument('--train_data_dir', type=str, default='../hw4_data/mini/train', help="Training images directory")
    parser.add_argument('--val_csv', type=str, default='../hw4_data/mini/val.csv', help="val images csv file")
    parser.add_argument('--val_data_dir', type=str, default='../hw4_data/mini/val', help="val images directory")
    parser.add_argument('--val_testcase_csv', type=str, default='../hw4_data/mini/val_testcase.csv', help="val test case csv")
    parser.add_argument('--ckpt_dir', default='ckpt/', type=str, help='Checkpoint path', required=False)
    parser.add_argument('--name', default='', type=str, help='Name for saving model')

    parser.add_argument('--num_epochs', type=int, default=100, help='number of total epochs')

    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok = True)
    return args

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs = args.num_epochs
    project_name = args.name
    

    train_dataset = MiniDataset(args.train_csv, args.train_data_dir)
    val_dataset = MiniDataset(args.val_csv, args.val_data_dir)

    train_loader = DataLoader(
        train_dataset,
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        batch_sampler=NShotTaskSampler(args.train_csv, args.episodes, args.N_way, args.N_shot, args.N_query))

    val_loader = DataLoader(
        val_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.val_testcase_csv))

    model = Protonet().to(device)
    parametric = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,  betas=(0.9, 0.98))
    if args.matching_fn == 'parametric':
        parametric = nn.Sequential(
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(400, 1)
        ).to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(parametric.parameters()), lr=1e-4,  betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        train_acc = []
        train_loss = []
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            support_input = data[:args.N_way * args.N_shot,:,:,:] 
            query_input   = data[args.N_way * args.N_shot:,:,:,:]

            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

            support = model(support_input)
            queries = model(query_input)
            prototypes = support.reshape(args.N_way, args.N_shot, -1).mean(dim=1)

            distances = pairwise_distances(queries, prototypes, args.matching_fn, parametric)

            loss = criterion(-distances, query_label)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            y_pred = (-distances).softmax(dim=1).max(1, keepdim=True)[1]
            train_acc.append(1. * y_pred.eq(query_label.view_as(y_pred)).sum().item() / len(query_label))
        scheduler.step()
        train_acc = np.array(train_acc)
        train_loss = sum(train_loss) / len(train_loss)
        print(f"Train accuracy {train_acc.mean():.4f} +- {train_acc.std():.4f}, loss {train_loss:.4f}")

        model.eval()
        valid_acc = []
        valid_loss = []

        with torch.no_grad():
            for b_idx, (data, target) in enumerate(val_loader):
                data = data.to(device)
                support_input = data[:args.N_way * args.N_shot,:,:,:] 
                query_input = data[args.N_way * args.N_shot:,:,:,:]

                label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
                query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

                support = model(support_input)
                queries = model(query_input)
                prototypes = support.reshape(args.N_way, args.N_shot, -1).mean(dim=1)

                distances = pairwise_distances(queries, prototypes, args.matching_fn, parametric)
                    
                valid_loss.append(criterion(-distances, query_label).item())
                y_pred = (-distances).softmax(dim=1).max(1, keepdim=True)[1]
                valid_acc.append(1. * y_pred.eq(query_label.view_as(y_pred)).sum().item() / len(query_label))

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = np.array(valid_acc)
        mean = valid_acc.mean()
        std = valid_acc.std()
        print(f"Valid accuracy {valid_acc.mean():.4f} +- {valid_acc.std():.4f}, loss {valid_loss:.4f}")

        to_save = {'model': model.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                 'parametric': None}
        if args.matching_fn == 'parametric':
            to_save['parametric'] = parametric.state_dict()

        fp = os.path.join(args.ckpt_dir, f'{epoch + 1}_{project_name}.pth')
        torch.save(to_save, fp)
        print(f"Saved model to {fp}")

    

if __name__=='__main__':
    same_seeds(123)
    args = parse_args()
    train(args)

    

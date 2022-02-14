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

def predict(args, data_loader, model, parametric = None):
    prediction_results = []
    episodic_acc = []

    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            support_input = data[:args.N_way * args.N_shot,:,:,:]
            query_input   = data[args.N_way * args.N_shot:,:,:,:]
            support_input = support_input.to(args.device)
            query_input = query_input.to(args.device)

            support = model(support_input)
            queries = model(query_input)

            support = support.reshape(args.N_shot, args.N_way, -1).mean(dim=0)

            logits = pairwise_distances(queries, support, args.matching_fn, parametric)
            pred = torch.argmin(logits, dim=1)
            prediction_results.append(pred.cpu().numpy().tolist())

    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', default='./hw4_data/mini/val.csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', default='./hw4_data/mini/val', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', default='./hw4_data/val_testcase.csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")
    parser.add_argument('--device', type=str, default='cuda', help='Device for evaluation')
    parser.add_argument('--matching_fn', default='parametric', type=str, help='distance matching function')

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # TODO: load your model
    state = torch.load(args.load)
    model = Protonet().cuda()
    model.load_state_dict(state['model'])
    model.eval()

    if args.matching_fn == 'parametric':
        parametric = nn.Sequential(
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(400, 1)
        ).cuda()
        parametric.load_state_dict(state['parametric'])
        parametric.eval()
    else:
        parametric = None
    
    prediction_results = predict(args, test_loader, model, parametric)

    # TODO: output your prediction to csv
    with open(args.output_csv, 'w') as f:
        line = ['episode_id'] + [f"query{i}" for i in range(args.N_way*args.N_query)]
        print(','.join(line), file = f)

        for i, prediction in enumerate(prediction_results):
            line = [str(i)] + [str(j) for j in prediction]
            print(','.join(line), file = f)

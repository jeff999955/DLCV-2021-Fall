import torch
import torch.nn as nn
from torch.utils.data import Sampler
import random
import numpy as np
import pandas as pd


def pairwise_distances(x, y, matching_fn='euclidean', parametric=None):
    m, n = x.shape[0], y.shape[0]
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    if matching_fn == 'euclidean':
        distances = (x.expand(m, n, -1) - y.expand(m, n, -1)).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        cosine_similarities = cos(x.expand(m, n, -1), y.expand(m, n, -1))

        return 1 - cosine_similarities
    elif matching_fn == 'parametric':
        x_exp = x.expand(m, n, -1).reshape(m*n, -1)
        y_exp = y.expand(m, n, -1).reshape(m*n, -1)
        
        distances = parametric(torch.cat([x_exp, y_exp], dim=-1))
        
        return distances.reshape(m, n)

    return -1

def same_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class NShotTaskSampler(Sampler):
    def __init__(self, csv_path, episodes_per_epoch, N_way, N_shot, N_query):
        self.data_df = pd.read_csv(csv_path)
        self.N_way = N_way
        self.N_shot = N_shot
        self.N_query = N_query
        self.episodes_per_epoch = episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []
            episode_classes = np.random.choice(self.data_df['label'].unique(), size=self.N_way, replace=False)

            support = []
            query = []

            for k in episode_classes:
                ind = self.data_df[self.data_df['label'] == k]['id'].sample(self.N_shot + self.N_query).values
                support = support + list(ind[:self.N_shot])
                query = query + list(ind[self.N_shot:])

            batch = support + query

            yield np.stack(batch)

    def __len__(self):
        return self.episodes_per_epoch


class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)
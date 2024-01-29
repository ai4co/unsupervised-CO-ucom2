import torch
from pathlib import Path
import re
import urllib.request
import random
import pickle

def get_random_dataset(num_items, num_sets, seed):
    # random.seed(seed)
    # dataset = []
    # for i in range(100):
    #     weights = [random.randint(1, 100) for _ in range(num_items)]
    #     sets = []
    #     for set_idx in range(num_sets):
    #         covered_items = random.randint(10, 30)
    #         sets.append(random.sample(range(num_items), covered_items))
    #     dataset.append((f'rand{i}', weights, sets))
    if seed == 1:
        with open(f"data/max_covering_{num_sets}_train.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        with open(f"data/max_covering_{num_sets}_test.pkl", "rb") as f:
            dataset = pickle.load(f)
    return dataset

def get_random_dataset_small(num_items, num_sets, seed):
    # random.seed(seed)
    # dataset = []
    # for i in range(100):
    #     weights = [random.randint(1, 100) for _ in range(num_items)]
    #     sets = []
    #     for set_idx in range(num_sets):
    #         covered_items = random.randint(10, 30)
    #         sets.append(random.sample(range(num_items), covered_items))
    #     dataset.append((f'rand{i}', weights, sets))
    
    with open(f"data/max_covering_{num_sets}_train.pkl", "rb") as f:
        dataset = pickle.load(f)    
    return dataset[:5]


def get_twitch_dataset():
    import math
    dataset = []
    languages = ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU']
    for language in languages:
        with open(f'data/twitch/{language}/musae_{language}_edges.csv') as f:
            edges = []
            node_ids = set()
            for e in f.readlines():
                e_str = e.strip().split(',')
                if e_str[0] == 'from' and e_str[1] == 'to':
                    continue
                n1, n2 = int(e_str[0]), int(e_str[1])
                edges.append((n1, n2))
                node_ids.add(n1)
                node_ids.add(n2)
        id_map = {n: i for i, n in enumerate(node_ids)}
        weights = [-1 for _ in node_ids]
        with open(f'data/twitch/{language}/musae_{language}_target.csv') as f:
            for line in f.readlines():
                line_str = line.strip().split(',')
                if line_str[0] == 'id':
                    continue
                weights[id_map[int(line_str[5])]] = math.floor(math.log(int(line_str[3]) + 1))
        assert min(weights) >= 0
        sets = [[] for _ in node_ids]
        for n1, n2 in edges:
            sets[id_map[n1]].append(id_map[n2])
        dataset.append((language, weights, sets))
    return dataset

def get_rail_dataset():
    with open("data/orlib/rail.data", "rb") as f:
        dataset = pickle.load(f)
    return dataset

# import
from ctypes import sizeof
import math
import pickle
import random
from collections import Counter, defaultdict
from itertools import combinations
import sys
from sys import getsizeof

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from networkx import gnp_random_graph
from numpy.random import permutation
from torch import Tensor, prod, tensor
from torch.nn import Linear
from tqdm.auto import tqdm, trange
import time
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, required=True)
parser.add_argument("--c", type=int, required=True)
args = parser.parse_args()

ds_name = args.ds
c = args.c
use_hard_cons = True
beta_high = 10000


p_res = Path(f"res_robust_coloring")
p_res.mkdir(exist_ok=True)
device = torch.device("cpu")

seed = 0
random.seed(seed)  # python random generator
np.random.seed(seed)  # numpy random generator

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

with open(f"data/{ds_name}.graph", "rb") as f:
    G = pickle.load(f)
uvw_list = sorted(
    [(u, v, w) for u, v, w in tqdm(G.edges(data="weight"), total=G.number_of_edges())],
    key=lambda x_: -x_[-1],
)
weight_list = [w for u, v, w in uvw_list]

num_hard_constraints = len(uvw_list) // 5 if use_hard_cons else 0
G_new = nx.Graph()
G_new.add_nodes_from(G.nodes())
nodes_orig = list(G.nodes())
pair2penalty = dict()

for i_uvw, (u, v, w) in tqdm(enumerate(uvw_list), total=len(uvw_list)):
    if i_uvw < num_hard_constraints:
        G_new.add_edge(u, v)
    else:
        pair2penalty[(min(u, v), max(u, v))] = np.log(1 / (1 - w))

G = G_new
n = G.number_of_nodes()
greedy_coloring = nx.coloring.greedy_color(G, strategy="largest_first")
n_greedy_coloring = len(set(greedy_coloring.values()))
print(f"greedy coloring uses {n_greedy_coloring} colors")
A = nx.adjacency_matrix(G, weight=None)
H_hard = torch.tensor(A.todense(), device=device).float()
H_soft = torch.zeros_like(H_hard)
for (i, j), p_ij in pair2penalty.items():
    H_soft[i, j] = H_soft[j, i] = p_ij
H_total = H_soft + H_hard * beta_high
nx_coloring_strategies = [
    "largest_first",
    # "random_sequential",
    # "smallest_last",
    # "independent_set",
    # "connected_sequential_bfs",
    # "connected_sequential_dfs",
    # "saturation_largest_first",
]

start = time.time()
time_budget = 60 * 5
res_gd = []
nx_strategy = "largest_first"
for _ in trange(10000):
    if time.time() - start > time_budget:
        break
    uvw_list_hard = np.array(uvw_list[:num_hard_constraints])
    uvw_list_soft = np.random.permutation(uvw_list[num_hard_constraints:])
    uvw_list_perm = np.concatenate((uvw_list_hard, uvw_list_soft), 0)
    def num_colors_margin(num_included_edges):
        G_collect = nx.from_edgelist(
            [(u, v) for u, v, _ in uvw_list_perm[:num_included_edges]]
        )
        greedy_coloring_ = nx.coloring.greedy_color(G_collect, strategy=nx_strategy)
        n_greedy_coloring_ = len(set(greedy_coloring_.values()))
        u_new, v_new, _ = uvw_list_perm[num_included_edges]
        G_collect.add_edge(u_new, v_new)
        greedy_coloring_new = nx.coloring.greedy_color(G_collect, strategy=nx_strategy)
        n_greedy_coloring_new = len(set(greedy_coloring_new.values()))
        if n_greedy_coloring_new <= c:
            return c - 1
        elif n_greedy_coloring_ > c:
            return c + 1
        else:
            return c

    def binary_search(low, high):
        x = c
        # Check base case
        if high >= low:

            mid = (high + low) // 2

            # If element is present at the middle itself
            if num_colors_margin(mid) == x:
                return mid

            # If element is smaller than mid, then it can only
            # be present in left subarray
            elif num_colors_margin(mid) > x:
                return binary_search(low, mid - 1)

            # Else the element can only be present in right subarray
            else:
                return binary_search(mid + 1, high)

        else:
            # Element is not present in the array
            return -1

    # max_possible_num_edges = binary_search(num_hard_constraints, G.number_of_edges())
    max_possible_num_edges = binary_search(num_hard_constraints, len(uvw_list_perm))
    if max_possible_num_edges == -1:
        max_possible_num_edges = binary_search(0, len(uvw_list_perm))
    if max_possible_num_edges == -1:
        max_possible_num_edges = c
    # print(max_possible_num_edges, num_hard_constraints, len(uvw_list_perm))
    G_for_max_coloring = nx.Graph()
    # G_for_max_coloring.add_nodes_from(G.nodes())
    G_for_max_coloring.add_nodes_from(nodes_orig)
    G_for_max_coloring.add_edges_from(
        [(u, v) for u, v, _ in uvw_list_perm[:max_possible_num_edges]]
    )
    
    max_greedy_coloring = nx.coloring.greedy_color(
        G_for_max_coloring, strategy=nx_strategy
    )
    # print(nx_strategy, max_possible_num_edges, len(set(max_greedy_coloring.values())))        
    
    cost_greedy = 0.0
    for (u, v), p_uv in pair2penalty.items():
        cu = max_greedy_coloring[u]
        cv = max_greedy_coloring[v]
        if cu == cv:
            cost_greedy += p_uv
    for u, v in G.edges():
        cu = max_greedy_coloring[u]
        cv = max_greedy_coloring[v]
        assert cu != cv
        if cu == cv:
            cost_greedy += beta_high
    # print(f"greedy can include at most {max_possible_num_edges} edges with {c} colors")    
    res_gd.append(cost_greedy)
use_time = time.time() - start
print(ds_name, c, use_time, min(res_gd))

with open(p_res / f"{ds_name}-{c}-DC.txt", "a+") as f:
    f.write(f"{min(res_gd)} {time.time() - start}\n")
print(min(res_gd), time.time() - start)

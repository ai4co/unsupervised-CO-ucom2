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

# greedy evaluation

P_I = beta_high

def decode(length, lehmer):
    """Return permutation for the given Lehmer Code and permutation length. Result permutation contains
    number from 0 to length-1.
    """
    # result = [(lehmer % factorial(length - i)) // factorial(length - 1 - i) for i in range(length)]
    result = lehmer + [0]
    used = [False] * length
    for i in range(length):
        counter = 0
        for j in range(length):
            if not used[j]:
                counter += 1
            if counter == result[i] + 1:
                result[i] = j
                used[j] = True
                break
    return result


def evaluate_color(node_order, i_node, node, color, node2color):
    cost = 0.0
    for i in range(i_node):
        node_i = node_order[i]
        if node2color[node_i] == color:
            if G.has_edge(node, node_i):
                return P_I
            else:
                cost += pair2penalty.get((min(node, node_i), max(node, node_i)), 0.0)
    return cost


def evaluate_all(node_order):
    node2color = dict()
    total_cost = 0.0
    for i_node, node in enumerate(node_order):
        if i_node < c:
            node2color[node] = i_node
        else:
            best_color = min(
                range(c),
                key=lambda x_: evaluate_color(node_order, i_node, node, x_, node2color),
            )
            total_cost += evaluate_color(
                node_order, i_node, node, best_color, node2color
            )
            node2color[node] = best_color
    # print_both(total_cost)
    return node2color, total_cost

def f(X):
    X = [int(x) for x in X]
    X_permutation = decode(n, X)
    node2color_, total_cost = evaluate_all(X_permutation)
    return node2color_, total_cost

start = time.time()
res_random = []
time_limit = 5 * 60
for _ in trange(10000):
    if time.time() - start >= time_limit:
        break
    xxx = [np.random.randint(0, n - i) for i in range(n - 1)]
    node2color, total_cost = f(xxx)
    res_random.append(total_cost)

with open(p_res / f"{ds_name}-{c}-random.txt", "a+") as f:
    f.write(f"{min(res_random)} {time.time() - start}\n")
print(min(res_random), time.time() - start)
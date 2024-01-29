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

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, required=True)
parser.add_argument("--c", type=int, required=True)
parser.add_argument("--npam", type=int, required=True)
args = parser.parse_args()

ds_name = args.ds
c = args.c
use_hard_cons = True
beta_high = 10000

p_res = Path(f"res_ours")
p_res.mkdir(exist_ok=True)

device = torch.device("cuda")

seed = 0
def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(0)

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

def loss_robust_coloring_individual(
    input_tensor: Tensor, reg_coef: float, with_softmax: bool = True
) -> Tensor:
    if with_softmax:
        input_tensor = torch.softmax(input_tensor, dim=-1)
    affinity_ = torch.bmm(input_tensor, input_tensor.transpose(1, 2))
    res = ((H_hard * reg_coef + H_soft) * affinity_).sum(dim=(-2, -1)) * 0.5
    return res

def loss_robust_coloring_for_derandomization(
    input_tensor: Tensor,
    cur_node: int,
    cur_pos: int,
    reg_coef: float,
    with_softmax: bool = True,
) -> Tensor:
    if with_softmax:
        input_tensor = torch.softmax(input_tensor, dim=-1)
    A_i = (H_hard * reg_coef + H_soft)[cur_node]
    P_x = input_tensor[:, :, cur_pos]
    return (A_i * P_x).sum(dim=-1)


lr = 0.1
beta = H_soft.max().item()
beta_high = 10000.0
num_epochs = 10
if args.npam <= 100:
    num_pams = args.npam
    num_rounds = 1
else:
    num_pams = 100
    num_rounds = args.npam // 100    

for i_round in range(num_rounds):
    xx = torch.ones((num_pams, n, c), device=device)
    xx += torch.rand_like(xx) * 0.1
    xx = torch.nn.Parameter(xx)
    optimizer = torch.optim.Adam([xx], lr=lr)
    A_train = H_hard * beta + H_soft
    A_reg = H_hard * beta_high + H_soft
    penalty_matrix = A_reg.cpu().numpy()
    
    def loss_robust_coloring(
        input_tensor: Tensor, reg_coef: float, with_softmax: bool = True
    ) -> Tensor:
        if with_softmax:
            input_tensor = torch.softmax(input_tensor, dim=-1)
        affinity_ = torch.bmm(input_tensor, input_tensor.transpose(1, 2))
        res = (A_train * affinity_).sum() * 0.5
        return res
    
    @torch.no_grad()
    def compute_obj_changes(cur_xx):
        xx_ = torch.softmax(cur_xx, dim=-1)  # (batch_size, n, c)
        # A_reg (n, n)
        # finally, the res should be (batch_size, n, c)
        # where res(x, i, j) = objective "contribution" when we color node i with color j in batch x
        # res(x, i, j) = \sum_k (A_reg[i, k] * xx_[x, k, j])
        res = torch.einsum("xkj, ik->xij", xx_, A_reg)
        # res_mean(x, i) = expected objective "contribution" with the current color distribution of node i
        # res_mean(x, i) = \sum_j res[x, i, j] * xx_[x, i, j] = (res * xx_).sum(dim=-1)
        res_mean = (res * xx_).sum(dim=-1).unsqueeze_(-1)
        # res - res_mean = objective change when we recolor node i with color j in batch x
        return res - res_mean
    
    @torch.no_grad()
    def compute_obj_changes_train(cur_xx):
        xx_ = torch.softmax(cur_xx, dim=-1)  # (batch_size, n, c)
        # A_reg (n, n)
        # finally, the res should be (batch_size, n, c)
        # where res(x, i, j) = objective "contribution" when we color node i with color j in batch x
        # res(x, i, j) = \sum_k (A_reg[i, k] * xx_[x, k, j])
        res = torch.einsum("xkj, ik->xij", xx_, A_train)
        # res_mean(x, i) = expected objective "contribution" with the current color distribution of node i
        # res_mean(x, i) = \sum_j res[x, i, j] * xx_[x, i, j] = (res * xx_).sum(dim=-1)
        res_mean = (res * xx_).sum(dim=-1).unsqueeze_(-1)
        # res - res_mean = objective change when we recolor node i with color j in batch x
        return res - res_mean
    
    essential_one = 1 - 1e-6
    
    
    @torch.no_grad()
    def compute_obj_changes_normalized(cur_xx):
        xx_ = torch.softmax(cur_xx, dim=-1)
        res = torch.einsum("xkj, ik->xij", xx_, A_reg)
        res_mean = (res * xx_).sum(dim=-1).unsqueeze_(-1)
        res_diff = res - res_mean
        var_diff = torch.abs(essential_one - xx_)
        return res_diff / var_diff
    
    def ratio2preSoftmax(final_ratio):
        assert 0 < final_ratio < 1
        ratio_ratio = final_ratio / (1 - final_ratio)
        return 0.5 * math.log(ratio_ratio * (c - 1))
    th_p99_sm = ratio2preSoftmax(essential_one)
    
    time_s = time.time()
    for i_epoch in trange(num_epochs):
        loss = loss_robust_coloring(xx, beta)
        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()
    time_e = time.time()
    print(f"training: used time = {time_e - time_s}")    
    
    
    def loss_robust_coloring_for_derandomization_all_colors(
        input_tensor: Tensor,
        cur_node: int,
        # cur_pos: int,
        reg_coef: float,
        with_softmax: bool = True,
    ) -> Tensor:
        if with_softmax:
            input_tensor = torch.softmax(input_tensor, dim=-1)
        # cost_hard = H_hard * reg_coef
        # A = cost_hard + H_soft
        # input_tensor has a shape of (batch_size, n, c)
        # A_i = A_reg[cur_node].reshape(-1, 1)
        A_i = A_reg[cur_node].unsqueeze(-1)
        # when cur_node is a single value, A_i [n, 1]
        # when cur_node is various, it is supposed to have a shape of [batch_size, n, 1]
        res = (A_i * input_tensor).sum(dim=1)
        # anyway, the product (A_i * input_tensor) should have a shape of [batch_size, n, c]
        # and the result should have a shape of [batch, c]
        return res
    
    
    def derandomization_faster(x, n_perm=None):
        # b = num_pams
        if n_perm is None:
            n_perm = range(n)
        x_detach = torch.softmax(x, dim=-1)
        for i in tqdm(n_perm):
            # consider i is a list of indices
            min_cond_obj = torch.full((num_pams,), torch.inf)
            j_best = torch.empty_like(min_cond_obj).long()
            obj_ij = loss_robust_coloring_for_derandomization_all_colors(
                x_detach, i, beta_high, with_softmax=False
            )  # shape = (b, c)
    
            j_best = torch.argmin(obj_ij, dim=1)
            x_detach[torch.arange(num_pams), i] = 0.0
            x_detach[torch.arange(num_pams), i, j_best] = 1.0
        return x_detach
    
    
    @torch.no_grad()
    def compute_obj_changes_final(cur_xx):
        # A_reg (n, n)
        # finally, the res should be (batch_size, n, c)
        # where res(x, i, j) = objective "contribution" when we color node i with color j in batch x
        # res(x, i, j) = \sum_k (A_reg[i, k] * xx_[x, k, j])
        res = torch.einsum("xkj, ik->xij", cur_xx, A_reg)
        # res_mean(x, i) = expected objective "contribution" with the current color distribution of node i
        # res_mean(x, i) = \sum_j res[x, i, j] * xx_[x, i, j] = (res * xx_).sum(dim=-1)
        res_mean = (res * cur_xx).sum(dim=-1).unsqueeze_(-1)
        # res - res_mean = objective change when we recolor node i with color j in batch x
        return res - res_mean
    
    @torch.no_grad()
    def compute_obj_changes_final_soft(cur_xx):
        # A_reg (n, n)
        # finally, the res should be (batch_size, n, c)
        # where res(x, i, j) = objective "contribution" when we color node i with color j in batch x
        # res(x, i, j) = \sum_k (A_reg[i, k] * xx_[x, k, j])
        res = torch.einsum("xkj, ik->xij", cur_xx, A_train)
        # res_mean(x, i) = expected objective "contribution" with the current color distribution of node i
        # res_mean(x, i) = \sum_j res[x, i, j] * xx_[x, i, j] = (res * xx_).sum(dim=-1)
        res_mean = (res * cur_xx).sum(dim=-1).unsqueeze_(-1)
        # res - res_mean = objective change when we recolor node i with color j in batch x
        return res - res_mean
    
    
    # after training
    final_res = []
    xx.detach_()
    xx = torch.softmax(xx, dim=-1)
    
    time_s = time.time()
    # greedily improve    
    obj_changes = compute_obj_changes_final_soft(xx)  # (b, n, c)
    while obj_changes.min() < 0:     
        best_indices = torch.argmin(
            obj_changes.reshape(num_pams, -1), dim=-1
        )  # (b) each entry in [n * c]
        best_nodes, best_colors = best_indices // c, best_indices % c
        range_pams = torch.arange(num_pams)
        xx[range_pams, best_nodes, :] = 0.0
        xx[range_pams, best_nodes, best_colors] = 1.0
        # print(torch.bitwise_and(xx != 0.0, xx != 1.0).sum(), end="\r")
        obj_changes = compute_obj_changes_final_soft(xx)  # (b, n, c)  
    
    obj_changes = compute_obj_changes_final(xx)  # (b, n, c)
    while obj_changes.min() < 0:        
        best_indices = torch.argmin(
            obj_changes.reshape(num_pams, -1), dim=-1
        )  # (b) each entry in [n * c]
        best_nodes, best_colors = best_indices // c, best_indices % c
        range_pams = torch.arange(num_pams)
        xx[range_pams, best_nodes, :] = 0.0
        xx[range_pams, best_nodes, best_colors] = 1.0
        # print(torch.bitwise_and(xx != 0.0, xx != 1.0).sum(), end="\r")
        obj_changes = compute_obj_changes_final(xx)  # (b, n, c)
    
    time_e = time.time()
    res_obj = loss_robust_coloring_individual(
        xx, beta_high, with_softmax=False
    )
    for res_ in res_obj.detach().cpu().numpy():
        final_res.append(res_)
    
    def stat_summary(raw_results):
        res_mean = np.mean(raw_results)
        res_std = np.std(raw_results)
        res_min = np.min(raw_results)
        print(f"{res_mean:.3f} +- {res_std:.3f}; {res_min:.3f}")
    
    stat_summary(final_res)
    print(f"derand after training: used time = {time_e - time_s}")

    # random init
    final_res = []
    xx = torch.rand((num_pams, n, c), device=device)
    xx = xx / xx.sum(-1).unsqueeze(-1)
    
    time_s = time.time()
    # greedily improve
    obj_changes = compute_obj_changes_final_soft(xx)  # (b, n, c)
    while obj_changes.min() < 0:     
        best_indices = torch.argmin(
            obj_changes.reshape(num_pams, -1), dim=-1
        )  # (b) each entry in [n * c]
        best_nodes, best_colors = best_indices // c, best_indices % c
        range_pams = torch.arange(num_pams)
        xx[range_pams, best_nodes, :] = 0.0
        xx[range_pams, best_nodes, best_colors] = 1.0
        # print(torch.bitwise_and(xx != 0.0, xx != 1.0).sum(), end="\r")
        obj_changes = compute_obj_changes_final_soft(xx)  # (b, n, c)    
    
    obj_changes = compute_obj_changes_final(xx)  # (b, n, c)
    while obj_changes.min() < 0:     
        best_indices = torch.argmin(
            obj_changes.reshape(num_pams, -1), dim=-1
        )  # (b) each entry in [n * c]
        best_nodes, best_colors = best_indices // c, best_indices % c
        range_pams = torch.arange(num_pams)
        xx[range_pams, best_nodes, :] = 0.0
        xx[range_pams, best_nodes, best_colors] = 1.0
        # print(torch.bitwise_and(xx != 0.0, xx != 1.0).sum(), end="\r")
        obj_changes = compute_obj_changes_final(xx)  # (b, n, c)
    time_e = time.time()
    res_obj = loss_robust_coloring_individual(
        xx, beta_high, with_softmax=False
    )
    
    for res_ in res_obj.detach().cpu().numpy():
        final_res.append(res_)
    
    def stat_summary(raw_results):
        res_mean = np.mean(raw_results)
        res_std = np.std(raw_results)
        res_min = np.min(raw_results)
        print(f"{res_mean:.3f} +- {res_std:.3f}; {res_min:.3f}")
    
    stat_summary(final_res)
    print(f"derand after random init: used time = {time_e - time_s}")

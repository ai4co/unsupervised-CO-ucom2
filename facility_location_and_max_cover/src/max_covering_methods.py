from copy import deepcopy
import random
from ortools.linear_solver import pywraplp
import numpy as np
import torch
import torch_geometric as pyg
import time

from tqdm import trange
from src.gumbel_sinkhorn_topk import gumbel_sinkhorn_topk
import src.perturbations as perturbations
import src.blackbox_diff as blackbox_diff
from src.poi_bin import *
from lap_solvers.lml import LML
from torch_geometric.utils import dropout_adj
from torch_geometric.loader import DataLoader

device_ = torch.device("cuda:0")

####################################
#        helper functions          #
####################################


def compute_objective(
    weights, sets, selected_sets, bipartite_adj=None, device=torch.device("cpu")
):
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float, device=device)

    if not isinstance(selected_sets, torch.Tensor):
        selected_sets = torch.tensor(selected_sets, device=device)

    if bipartite_adj is None:
        bipartite_adj = torch.zeros(
            (len(sets), len(weights)), dtype=torch.float, device=device
        )
        for _i, _set in enumerate(sets):
            bipartite_adj[_i, _set] = 1

    selected_items = bipartite_adj[selected_sets, :].sum(dim=-2).clamp_(0, 1)
    return torch.matmul(selected_items, weights)


# Main obj, Wang et al, ICLR'23
def compute_obj_differentiable(
    weights, sets, latent_probs, bipartite_adj=None, device=torch.device("cpu")
):
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float, device=device)
    if not isinstance(latent_probs, torch.Tensor):
        latent_probs = torch.tensor(latent_probs, device=device)
    if bipartite_adj is None:
        bipartite_adj = torch.zeros(
            (len(sets), len(weights)), dtype=torch.float, device=device
        )
        for _i, _set in enumerate(sets):
            bipartite_adj[_i, _set] = 1
    selected_items = torch.clamp_max(torch.matmul(latent_probs, bipartite_adj), 1)
    return torch.matmul(selected_items, weights), bipartite_adj


# Main obj from UCom2
def compute_obj_differentiable_fixed(
    weights, sets, latent_probs, bipartite_adj=None, device=torch.device("cpu")
):
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float, device=device)
    # weights: [n]
    if not isinstance(latent_probs, torch.Tensor):
        latent_probs = torch.tensor(latent_probs, device=device)
    # latent_probs: [m]
    if bipartite_adj is None:
        bipartite_adj = torch.zeros(
            (len(sets), len(weights)), dtype=torch.float, device=device
        )
        for _i, _set in enumerate(sets):
            bipartite_adj[_i, _set] = 1
    # bipartite_adj: [m, n]
    selected_items = 1 - (1 - latent_probs.unsqueeze(-1) * bipartite_adj).prod(dim=-2)
    # selected_items = torch.clamp_max(torch.matmul(latent_probs, bipartite_adj), 1)
    # return torch.matmul(selected_items, weights), bipartite_adj
    return (selected_items * weights).sum(), bipartite_adj


def compute_obj_differentiable_prepare(
    weights, sets, latent_probs, bipartite_adj=None, device=torch.device("cpu")
):
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float, device=device)
    # latent_probs: [m]
    if bipartite_adj is None:
        bipartite_adj = torch.zeros(
            (len(sets), len(weights)), dtype=torch.float, device=device
        )
        for _i, _set in enumerate(sets):
            bipartite_adj[_i, _set] = 1
    return weights, bipartite_adj


def compute_bipartite_adj(weights, sets, device):
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float, device=device)
    # weights: [n]
    bipartite_adj = torch.zeros(
        (len(sets), len(weights)), dtype=torch.float, device=device
    )
    for _i, _set in enumerate(sets):
        bipartite_adj[_i, _set] = 1
    return bipartite_adj


class BipartiteData(pyg.data.Data):
    def __init__(self, edge_index, x_src, x_dst):
        super(BipartiteData, self).__init__()
        self.edge_index = edge_index
        self.x1 = x_src
        self.x2 = x_dst

    # def __inc__(self, key, value):
    #     if key == "edge_index":
    #         return torch.tensor([[self.x1.size(0)], [self.x2.size(0)]])
    #     else:
    #         return super(BipartiteData, self).__inc__(key, value)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.x1.size(0)], [self.x2.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


def build_graph_from_weights_sets(weights, sets, device=torch.device("cpu")):
    x_src = torch.ones(len(sets), 1, device=device)
    x_tgt = torch.tensor(weights, dtype=torch.float, device=device).unsqueeze(-1)
    index_1, index_2 = [], []
    for set_idx, set_items in enumerate(sets):
        for set_item in set_items:
            index_1.append(set_idx)
            index_2.append(set_item)
    edge_index = torch.tensor([index_1, index_2], device=device)
    return BipartiteData(edge_index, x_src, x_tgt)


#################################################
#         Learning Max-kVC Methods              #
#################################################


class GNNModel(torch.nn.Module):
    # Max covering model (3-layer Bipartite SageConv)
    def __init__(self):
        super(GNNModel, self).__init__()
        self.gconv1_w2s = pyg.nn.SAGEConv((1, 1), 16)
        self.gconv1_s2w = pyg.nn.SAGEConv((1, 1), 16)
        self.gconv2_w2s = pyg.nn.SAGEConv((16, 16), 16)
        self.gconv2_s2w = pyg.nn.SAGEConv((16, 16), 16)
        self.gconv3_w2s = pyg.nn.SAGEConv((16, 16), 16)
        self.gconv3_s2w = pyg.nn.SAGEConv((16, 16), 16)
        self.fc = torch.nn.Linear(16, 1)

    def forward(self, g, return_log=False):
        # assert type(g) == BipartiteData
        reverse_edge_index = torch.stack((g.edge_index[1], g.edge_index[0]), dim=0)
        new_x1 = torch.relu(self.gconv1_w2s((g.x2, g.x1), reverse_edge_index))
        new_x2 = torch.relu(self.gconv1_s2w((g.x1, g.x2), g.edge_index))
        x1, x2 = new_x1, new_x2
        new_x1 = torch.relu(self.gconv2_w2s((x2, x1), reverse_edge_index))
        new_x2 = torch.relu(self.gconv2_s2w((x1, x2), g.edge_index))
        x1, x2 = new_x1, new_x2
        new_x1 = torch.relu(self.gconv3_w2s((x2, x1), reverse_edge_index))
        new_x2 = torch.relu(self.gconv3_s2w((x1, x2), g.edge_index))
        x = self.fc(new_x1).squeeze(-1)
        if return_log:
            return torch.sigmoid(x), x
        return torch.sigmoid(x)


def egn_max_covering(
    weights, sets, max_covering_items, model, egn_beta, random_trials=0, time_limit=-1
):
    prev_time = time.time()
    graph = build_graph_from_weights_sets(weights, sets, device_)
    graph.ori_x1 = graph.x1.clone()
    graph.ori_x2 = graph.x2.clone()
    best_objective = 0
    best_top_k_indices = None
    bipartite_adj = None
    obj_list = []
    time_list = []
    for _ in trange(random_trials if random_trials > 0 else 1):
        if time_limit > 0 and time.time() - prev_time > time_limit:
            break
        if random_trials > 0:
            graph_copy = graph.clone()
            graph_copy.x1 = graph_copy.x1 + torch.randn_like(graph.x1) * 0.2
            graph_copy.x2 = graph_copy.x2 + torch.randn_like(graph.x2) * 0.2
            graph_copy.edge_index, graph_copy.edge_attr = dropout_adj(
                graph.edge_index, graph.edge_attr, p=0.2
            )
        else:
            graph_copy = graph.clone()
        probs = model(graph_copy).detach()
        dist_probs, probs_argsort = torch.sort(probs, descending=True)
        selected_items = 0
        for prob_idx in probs_argsort:
            if selected_items >= max_covering_items:
                probs[prob_idx] = 0
                continue
            probs_0 = probs.clone()
            probs_0[prob_idx] = 0
            probs_1 = probs.clone()
            probs_1[prob_idx] = 1
            constraint_conflict_0 = torch.relu(probs_0.sum() - max_covering_items)
            constraint_conflict_1 = torch.relu(probs_1.sum() - max_covering_items)
            obj_0, bipartite_adj = compute_obj_differentiable(
                weights, sets, probs_0, bipartite_adj, device=probs.device
            )
            obj_0 = obj_0 - egn_beta * constraint_conflict_0
            obj_1, bipartite_adj = compute_obj_differentiable(
                weights, sets, probs_1, bipartite_adj, device=probs.device
            )
            obj_1 = obj_1 - egn_beta * constraint_conflict_1
            if obj_0 <= obj_1:
                probs[prob_idx] = 1
                selected_items += 1
            else:
                probs[prob_idx] = 0
        top_k_indices = probs.nonzero().squeeze()
        objective = compute_objective(
            weights, sets, top_k_indices, bipartite_adj, device=probs.device
        ).item()
        if objective > best_objective:
            best_objective = objective
            best_top_k_indices = top_k_indices
        obj_list.append(best_objective)
        time_list.append(time.time() - prev_time)
    # return best_objective, best_top_k_indices, time.time() - prev_time
    return obj_list, time_list


def egn_pb_greedy_max_covering(
    weights, sets, max_covering_items, model, egn_beta, random_trials=0, time_limit=-1
):
    prev_time = time.time()
    graph = build_graph_from_weights_sets(weights, sets, device_)
    graph.ori_x1 = graph.x1.clone()
    graph.ori_x2 = graph.x2.clone()
    best_objective = 0
    best_top_k_indices = None
    bipartite_adj = None

    # if time_limit > 0 and time.time() - prev_time > time_limit:
    #     break

    for i_trial in range(random_trials):
        if i_trial > 0:
            graph.x1 = graph.ori_x1 + torch.randn_like(graph.ori_x1) / 100
            graph.x2 = graph.ori_x2 + torch.randn_like(graph.ori_x2) / 100

        probs = model(graph).detach()
        # print(probs.sum(), max_covering_items)
        # dist_probs, probs_argsort = torch.sort(probs, descending=True)

        improved = True
        improved_times = 0
        while improved:
            improved = False
            vx2result = torch.empty(probs.shape[0], 2).to(probs.device)
            for i in range(probs.shape[0]):
                probs_0 = probs.clone()
                k_diff = torch.abs(
                    torch.arange(probs_0.shape[0] + 1, device=probs.device)
                    - max_covering_items
                )

                probs_0[i] = 0
                card_dist = pmf_poibin_vec(
                    probs_0, probs.device, use_normalization=False
                )
                constraint_conflict_0 = (card_dist * k_diff).sum()
                # obj_0, bipartite_adj = compute_obj_differentiable(weights, sets, probs_0, bipartite_adj, device=probs.device)
                obj_0, bipartite_adj = compute_obj_differentiable_fixed(
                    weights, sets, probs_0, bipartite_adj, device=probs.device
                )
                vx2result[i, 0] = obj_0 - 10000.0 * constraint_conflict_0

                probs_0[i] = 1
                card_dist = pmf_poibin_vec(
                    probs_0, probs.device, use_normalization=False
                )
                constraint_conflict_1 = (card_dist * k_diff).sum()
                # obj_1, bipartite_adj = compute_obj_differentiable(weights, sets, probs_0, bipartite_adj, device=probs.device)
                obj_1, bipartite_adj = compute_obj_differentiable_fixed(
                    weights, sets, probs_0, bipartite_adj, device=probs.device
                )
                vx2result[i, 1] = obj_1 - 10000.0 * constraint_conflict_1

            best_index = torch.argmax(vx2result)
            # print(best_index)
            best_v, best_x = best_index // 2, best_index % 2
            if probs[best_v] != best_x:
                probs[best_v] = best_x
                improved = True
                improved_times += 1
                print(improved_times, end="\r")

            # top_k_indices = probs.nonzero().squeeze()
            # print(probs)
            top_k_indices = torch.topk(
                probs, max_covering_items, dim=-1
            ).indices.flatten()

            objective = compute_objective(
                weights, sets, top_k_indices, bipartite_adj, device=probs.device
            ).item()
            if objective > best_objective:
                best_objective = objective
                best_top_k_indices = top_k_indices
    return best_objective, best_top_k_indices, time.time() - prev_time


@torch.no_grad()
def egn_pb_greedy_faster_max_covering(
    weights,
    sets,
    max_covering_items,
    model,
    egn_beta,
    random_trials=0,
    time_limit=-1,
):
    prev_time = time.time()
    graph = build_graph_from_weights_sets(weights, sets, device_)
    weights = torch.tensor(weights, dtype=torch.float, device=device_)
    graph.ori_x1 = graph.x1.clone()
    graph.ori_x2 = graph.x2.clone()
    best_objective = None
    best_top_k_indices = None
    bipartite_adj = None

    for i_trial in range(random_trials if random_trials > 0 else 1):
        if i_trial > 0:
            graph.x1 = graph.ori_x1 + torch.randn_like(graph.ori_x1) * 0.1
            graph.x2 = graph.ori_x2 + torch.randn_like(graph.ori_x2) * 0.1
        probs = model(graph).detach()

        # breakpoint()
        improved = True
        improved_times = 0

        # latent_probs: [m]
        if bipartite_adj is None:
            bipartite_adj = torch.zeros(
                (len(sets), len(weights)), dtype=torch.float, device=probs.device
            )
            for _i, _set in enumerate(sets):
                bipartite_adj[_i, _set] = 1

        vx2result = torch.empty(probs.shape[0], 2).to(probs.device)
        # card_dist_orig = pmf_poibin_vec(probs, probs.device, use_normalization=False)
        while improved:
            improved = False
            # cur_time = time.time()
            card_dist_orig = pmf_poibin_vec(
                probs, probs.device, use_normalization=False
            )
            # print(f"card_dist_orig takes {time.time() - cur_time}")

            # cur_time = time.time()
            # bipartite_adj: [m, n]
            selected_items = 1 - (1 - probs.unsqueeze(-1) * bipartite_adj).prod(dim=0)

            # p2, constriant 2
            xx_ = probs.unsqueeze(0)
            pmf_cur = card_dist_orig.unsqueeze(0)
            n_nodes = probs.shape[-1]
            k = max_covering_items
            # [0, 0.5]
            term_1a = (1.0 / (1 - xx_)).unsqueeze(-1)
            # (1 - X_v)^{-1}; [b, n, 1]; term_1a[ib, iv] = (1 - X^{(b)}_v)^{-1}
            q_flip = pmf_cur.flip(1)
            q_roll_stack = torch.tril(
                torch.as_strided(
                    q_flip.repeat(1, 2),
                    (q_flip.shape[0], q_flip.shape[1], q_flip.shape[1]),
                    (q_flip.shape[1] * 2, 1, 1),
                ).flip(1)
            )[:, :n_nodes]
            # q_roll_stack[ib, i] = (q_i, q_{i-1}, ..., q_0, 0, ..., 0), 0 <= i < n; [b, n, n + 1]
            term_2a = (xx_ / (xx_ - 1.0)).unsqueeze(-1) ** torch.arange(
                n_nodes + 1, device=probs.device
            )
            # term_2a[ib, iv, i] = (X^{(b)}_v / (X^{(b)}_v - 1))^{i}; [b, n, n + 1]
            term_2a = torch.einsum(
                "bix, bvx -> bvi", q_roll_stack, term_2a
            )  # [b, n, n]
            res_case_1 = term_1a * term_2a

            # [0.5, 1]
            term_1b = (1.0 / xx_).unsqueeze(-1)
            q = pmf_cur
            q_roll_stack = torch.tril(
                torch.as_strided(
                    q.repeat(1, 2),
                    (q.shape[0], q.shape[1], q.shape[1]),
                    (q.shape[1] * 2, 1, 1),
                ).flip(1)
            )[:, :n_nodes].flip(1)
            # q_roll_stack[ib, i] = (q_{i + 1}, q_{i + 2}, ..., q_n, 0, ..., 0), 0 <= i < n; [b, n, n + 1]
            term_2b = ((xx_ - 1.0) / xx_).unsqueeze(-1) ** torch.arange(
                n_nodes + 1, device=probs.device
            )
            term_2b = torch.einsum("bix, bvx -> bvi", q_roll_stack, term_2b)
            res_case_2 = term_1b * term_2b
            tilde_q = torch.where(
                (res_case_1 <= 1.01) & (res_case_1 >= -0.01), res_case_1, res_case_2
            )
            tilde_q.clamp_(0.0, 1.0)
            # k_diff = torch.relu(torch.arange(n_nodes + 1, device=probs.device) - k)  # [n]
            k_diff_remove = torch.relu(
                torch.arange(n_nodes, device=probs.device) - k
            )  # [n]
            pmf_new = tilde_q[:, :, :n_nodes]  # [b, n, n]
            k_diff_add = torch.relu(
                torch.arange(n_nodes, device=probs.device) + 1 - k
            )  # [n]
            dol_remove_p2 = (pmf_new * k_diff_remove).sum(dim=-1)  # [b, n]
            dol_add_p2 = (pmf_new * k_diff_add).sum(dim=-1)  # [b, n]

            # print(f"cardinalty incremental takes {time.time() - cur_time}")

            # cur_time = time.time()
            vx2result.zero_()
            vx2result[:, 0] = -egn_beta * dol_remove_p2[0]
            vx2result[:, 1] = -egn_beta * dol_add_p2[0]

            C_ = (
                1 - probs.unsqueeze(-1) * bipartite_adj
            )  # [m, n]; C[j, i] = 1 - X_j * A_{ji}
            D_ = 1 - bipartite_adj  # [m, n]; D[j, i] = 1 - A_{ji}
            # [m, n]: selected_items_remove[j, i] = prob[i selected] when X_j <- 0
            selected_items_remove = 1 - (1 - selected_items) / (C_)
            # [m, n]: selected_items_add[j, i] = prob[i selected] when X_j <- 1
            selected_items_add = 1 - (1 - selected_items) * D_ / (C_)
            vx2result[:, 0] += (selected_items_remove * weights).sum(-1)
            vx2result[:, 1] += (selected_items_add * weights).sum(-1)

            # print(f"main obj incremental takes {time.time() - cur_time}")

            # for i in range(probs.shape[0]):
            #     probs_0 = probs.clone()
            #     probs_0[i] = 0
            #     obj_0, bipartite_adj = compute_obj_differentiable_fixed(weights, sets, probs_0, bipartite_adj, device=probs.device)
            #     vx2result[i, 0] = obj_0 + vx2result[i, 0]

            #     probs_0[i] = 1
            #     obj_1, bipartite_adj = compute_obj_differentiable_fixed(weights, sets, probs_0, bipartite_adj, device=probs.device)
            #     vx2result[i, 1] = obj_1 + vx2result[i, 1]
            # cur_time = time.time()
            best_index = torch.argmax(vx2result)
            # print(f"argmax takes {time.time() - cur_time}")
            # print(best_index)
            # cur_time = time.time()
            best_v, best_x = best_index // 2, best_index % 2
            # cur_time = time.time()
            if (torch.abs(probs[best_v] - best_x) >= 0.01).item():
                probs[best_v] = 0.999 if best_x else 0.001
                improved = True
                improved_times += 1
            print(improved_times, end="\r")

        top_k_indices = torch.topk(probs, max_covering_items, dim=-1).indices.flatten()
        objective = compute_objective(
            weights, sets, top_k_indices, bipartite_adj, device=probs.device
        ).item()
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_top_k_indices = top_k_indices
    return best_objective, best_top_k_indices, time.time() - prev_time


@torch.no_grad()
def egn_pb_greedy_faster_max_covering_full_truncated(
    weights,
    sets,
    graph,
    bipartite_adj,
    max_covering_items,
    model,
    egn_beta,
    random_trials=1,
    num_rounds=1,
    time_limit=-1,
    noise_scale=0.1,
    stru_noise=0.05,
    s_max=-1,
    zero_start=False,
):
    prev_time = time.time()
    weights = torch.tensor(weights, dtype=torch.float, device=device_)
    # graph = build_graph_from_weights_sets(weights, sets, device_)
    # bipartite_adj = None
    # graph.ori_x1 = graph.x1.clone()
    # graph.ori_x2 = graph.x2.clone()
    best_objective = None
    best_top_k_indices = None
    res_list = []

    for i_round in range(num_rounds):
        graph_list = []
        for i_trial in range(random_trials):
            if i_round == 0 and i_trial == 0:
                graph_copy = graph.clone()
                graph_list.append(graph_copy)
            else:
                graph_copy = graph.clone()
                graph_copy.x1 = graph_copy.x1 + torch.randn_like(graph.x1) * noise_scale
                graph_copy.x2 = graph_copy.x2 + torch.randn_like(graph.x2) * noise_scale
                if stru_noise > 0:
                    graph_copy.edge_index, graph_copy.edge_attr = dropout_adj(
                        graph.edge_index, graph.edge_attr, p=stru_noise
                    )
                graph_list.append(graph_copy)
        loader = DataLoader(graph_list, batch_size=len(graph_list))
        graph_batch = next(iter(loader))
        probs_batch = model(graph_batch).detach()

        n_nodes = graph.x1.shape[0]
        probs_matrix = probs_batch.view(-1, n_nodes)

        if zero_start:
            probs_matrix = torch.zeros_like(probs_matrix)

        k = max_covering_items
        if s_max <= 0:
            s_max = n_nodes + 1

        for round_hard in range(2):
            if round_hard:
                egn_beta = n_nodes * 100
            improved = [True for _ in range(random_trials)]
            while any(improved):
                if time_limit > 0 and time.time() - prev_time > time_limit:
                    break
                card_dist_orig = pmf_poibin(
                    probs_matrix, probs_matrix.device, use_normalization=False
                )
                # bipartite_adj: [m, n]

                xx_ = probs_matrix.unsqueeze(-1)
                yy_ = 1.0 - xx_

                C_ = 1 - xx_ * bipartite_adj

                selected_items = 1 - C_.prod(dim=-2)
                # p2, constriant 2
                pmf_cur = card_dist_orig
                # [0, 0.5]
                # term_1a = (1.0 / (1 - xx_)).unsqueeze(-1)
                # (1 - X_v)^{-1}; [b, n, 1]; term_1a[ib, iv] = (1 - X^{(b)}_v)^{-1}
                q_flip = pmf_cur.flip(1)
                q_roll_stack = torch.tril(
                    torch.as_strided(
                        q_flip.repeat(1, 2),
                        (q_flip.shape[0], q_flip.shape[1], q_flip.shape[1]),
                        (q_flip.shape[1] * 2, 1, 1),
                    ).flip(1)
                )[:, :n_nodes, :s_max]

                # q_roll_stack[ib, i] = (q_i, q_{i-1}, ..., q_0, 0, ..., 0), 0 <= i < n; [b, n, n + 1]
                term_2a = (-xx_ / yy_) ** torch.arange(
                    s_max,
                    device=probs_matrix.device
                    # n_nodes + 1, device=probs_matrix.device
                )
                # term_2a[ib, iv, i] = (X^{(b)}_v / (X^{(b)}_v - 1))^{i}; [b, n, n + 1]
                # term_2a = torch.einsum("bix, bvx -> bvi", q_roll_stack, term_2a)  # [b, n, n]
                term_2a = term_2a @ q_roll_stack.transpose(1, 2)  # [b, n, n]
                # res_case_1 = term_1a * term_2a
                res_case_1 = term_2a / yy_
                # del term_2a

                # [0.5, 1]
                # term_1b = (1.0 / xx_).unsqueeze(-1)
                q = pmf_cur
                q_roll_stack = torch.tril(
                    torch.as_strided(
                        q.repeat(1, 2),
                        (q.shape[0], q.shape[1], q.shape[1]),
                        (q.shape[1] * 2, 1, 1),
                    ).flip(1)
                )[:, :n_nodes].flip(1)[..., :s_max]
                # q_roll_stack[ib, i] = (q_{i + 1}, q_{i + 2}, ..., q_n, 0, ..., 0), 0 <= i < n; [b, n, n + 1]
                term_2b = (-yy_ / xx_) ** torch.arange(
                    s_max,
                    device=probs_matrix.device
                    # n_nodes + 1, device=probs_matrix.device
                )
                # term_2b = torch.einsum("bix, bvx -> bvi", q_roll_stack, term_2b)
                term_2b = term_2b @ q_roll_stack.transpose(1, 2)  # [b, n, n]

                # res_case_2 = term_1b * term_2b
                res_case_2 = term_2b / xx_
                # del term_2b
                tilde_q = torch.where(xx_ <= 0.5, res_case_1, res_case_2)
                tilde_q.clamp_(0.0, 1.0)

                n_nodes_arange = torch.arange(n_nodes, device=probs_matrix.device) - k
                k_diff_remove = torch.abs(n_nodes_arange)
                k_diff_add = torch.abs(n_nodes_arange + 1)

                pmf_new = tilde_q[..., :n_nodes]  # [b, n, n]

                dol_remove_p2 = (pmf_new * k_diff_remove).sum(dim=-1)  # [b, n]
                dol_add_p2 = (pmf_new * k_diff_add).sum(dim=-1)  # [b, n]
                vx2result = torch.empty(
                    probs_matrix.shape[0], probs_matrix.shape[1], 2
                ).to(probs_matrix.device)
                vx2result[..., 0] = -dol_remove_p2 * egn_beta
                vx2result[..., 1] = -dol_add_p2 * egn_beta

                # C_ = (
                #     1 - probs_matrix.unsqueeze(-1) * bipartite_adj
                # )  # [b, m, n]; C[ib, j, i] = 1 - X_{ib; j} * A_{ji}
                D_ = 1 - bipartite_adj  # [m, n]; D[j, i] = 1 - A_{ji}
                # [b, m, n]: selected_items_remove[j, i] = prob[i selected] when X_j <- 0
                E_ = (1 - selected_items.unsqueeze(-2)) / (C_)
                selected_items_remove = 1 - E_
                # [b, m, n]: selected_items_add[j, i] = prob[i selected] when X_j <- 1
                selected_items_add = 1 - D_ * E_
                vx2result[..., 0] += (selected_items_remove * weights).sum(-1)
                vx2result[..., 1] += (selected_items_add * weights).sum(-1)

                vx2result_flatten = vx2result.view(random_trials, -1)
                k_top = 1
                vx2result_flatten_topk = torch.topk(
                    vx2result_flatten, k_top, largest=True
                ).indices
                for i_input in range(random_trials):
                    if not improved[i_input]:
                        continue
                    improved[i_input] = False
                    for best_index in vx2result_flatten_topk[i_input]:
                        if improved[i_input]:
                            break
                        best_v, best_x = best_index // 2, best_index % 2
                        # if torch.abs(probs_matrix[i_input, best_v] - best_x) >= 0.01:
                        if torch.abs(probs_matrix[i_input, best_v] - best_x) >= 0.01:
                            probs_matrix[i_input, best_v] = 0.001 + best_x * 0.998
                            improved[i_input] = True
        top_k_indices = torch.topk(probs_matrix, k, dim=-1).indices
        objective = compute_objective(
            weights, sets, top_k_indices, bipartite_adj, device=probs_matrix.device
        )
        best_idx = torch.argmax(objective)
        top_k_indices = top_k_indices[best_idx]
        objective = objective[best_idx].item()
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_top_k_indices = top_k_indices
        res_list.append((best_objective, best_top_k_indices, time.time() - prev_time))
    # return best_objective, best_top_k_indices, time.time() - prev_time
    return res_list


def sinkhorn_max_covering_fast(
    weights,
    sets,
    max_covering_items,
    model,
    sample_num,
    noise,
    tau,
    sk_iters,
    opt_iters,
    sample_num2=None,
    noise2=None,
    verbose=True,
    iters_ratio=1.0,
):
    prev_time = time.time()
    graph = build_graph_from_weights_sets(weights, sets, device_)
    latent_vars = model(graph).detach()
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=0.1)
    bipartite_adj = None
    best_obj = 0
    best_top_k_indices = []
    best_found_at_idx = -1
    if (
        type(noise) == list
        and type(tau) == list
        and type(sk_iters) == list
        and type(opt_iters) == list
    ):
        iterable = zip(noise, tau, sk_iters, opt_iters)
    else:
        iterable = zip([noise], [tau], [sk_iters], [opt_iters])

    for noise, tau, sk_iters, opt_iters in iterable:
        for train_idx in range(int(opt_iters * iters_ratio)):
            gumbel_weights_float = torch.sigmoid(latent_vars)
            # noise = 1 - 0.75 * train_idx / 1000
            top_k_indices, probs = gumbel_sinkhorn_topk(
                gumbel_weights_float,
                max_covering_items,
                max_iter=sk_iters,
                tau=tau,
                sample_num=sample_num,
                noise_fact=noise,
                return_prob=True,
            )
            obj, bipartite_adj = compute_obj_differentiable(
                weights, sets, probs, bipartite_adj, probs.device
            )
            (-obj).mean().backward()
            # if train_idx % 10 == 0 and verbose:
            #     print(
            #         f"idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}"
            #     )
            if sample_num2 is not None and noise2 is not None:
                top_k_indices, probs = gumbel_sinkhorn_topk(
                    gumbel_weights_float,
                    max_covering_items,
                    max_iter=sk_iters,
                    tau=tau,
                    sample_num=sample_num2,
                    noise_fact=noise2,
                    return_prob=True,
                )
            obj = compute_objective(
                weights, sets, top_k_indices, bipartite_adj, device=probs.device
            )
            best_idx = torch.argmax(obj)
            max_obj, top_k_indices = obj[best_idx], top_k_indices[best_idx]
            if max_obj > best_obj:
                best_obj = max_obj
                best_top_k_indices = top_k_indices
                best_found_at_idx = train_idx
            if train_idx % 10 == 0 and verbose:
                print(
                    # f"idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}"
                    f"idx {train_idx} {obj.max():.2f} {obj.mean():.2f} {best_obj:.2f} {best_found_at_idx} {(time.time() - prev_time):.4f}"
                )
            optimizer.step()
            optimizer.zero_grad()
    return best_obj, best_top_k_indices


@torch.no_grad()
def sinkhorn_max_covering_fast_noTTO(
    weights,
    sets,
    max_covering_items,
    model,
    sample_num,
    noise,
    tau,
    sk_iters,
    opt_iters,
    sample_num2=None,
    noise2=None,
    verbose=True,
    iters_ratio=1.0,
):
    prev_time = time.time()
    graph = build_graph_from_weights_sets(weights, sets, device_)
    latent_vars = model(graph).detach()
    bipartite_adj = None
    best_obj = 0
    best_top_k_indices = []
    best_found_at_idx = -1
    if (
        type(noise) == list
        and type(tau) == list
        and type(sk_iters) == list
        and type(opt_iters) == list
    ):
        iterable = zip(noise, tau, sk_iters, opt_iters)
    else:
        iterable = zip([noise], [tau], [sk_iters], [opt_iters])
    for noise, tau, sk_iters, opt_iters in iterable:
        for train_idx in range(int(opt_iters * iters_ratio)):
            gumbel_weights_float = torch.sigmoid(latent_vars)
            # noise = 1 - 0.75 * train_idx / 1000
            top_k_indices, probs = gumbel_sinkhorn_topk(
                gumbel_weights_float,
                max_covering_items,
                max_iter=sk_iters,
                tau=tau,
                sample_num=sample_num,
                noise_fact=noise,
                return_prob=True,
            )
            obj, bipartite_adj = compute_obj_differentiable(
                weights, sets, probs, bipartite_adj, probs.device
            )
            # (-obj).mean().backward()
            # if train_idx % 10 == 0 and verbose:
            #     print(
            #         f"idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}"
            #     )
            if sample_num2 is not None and noise2 is not None:
                top_k_indices, probs = gumbel_sinkhorn_topk(
                    gumbel_weights_float,
                    max_covering_items,
                    max_iter=sk_iters,
                    tau=tau,
                    sample_num=sample_num2,
                    noise_fact=noise2,
                    return_prob=True,
                )
            obj = compute_objective(
                weights, sets, top_k_indices, bipartite_adj, device=probs.device
            )
            best_idx = torch.argmax(obj)
            max_obj, top_k_indices = obj[best_idx], top_k_indices[best_idx]
            if max_obj > best_obj:
                best_obj = max_obj
                best_top_k_indices = top_k_indices
                best_found_at_idx = train_idx
            if train_idx % 10 == 0 and verbose:
                print(
                    # f"idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}"
                    f"idx {train_idx} {obj.max():.2f} {obj.mean():.2f} {best_obj:.2f} {best_found_at_idx} {(time.time() - prev_time):.4f}"
                )
            # optimizer.step()
            # optimizer.zero_grad()
    return best_obj, best_top_k_indices


def sinkhorn_max_covering(
    weights,
    sets,
    max_covering_items,
    model,
    sample_num,
    noise,
    tau,
    sk_iters,
    opt_iters,
    sample_num2=None,
    noise2=None,
    verbose=True,
):
    graph = build_graph_from_weights_sets(weights, sets, device_)
    latent_vars = model(graph).detach()
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=0.1)
    bipartite_adj = None
    best_obj = 0
    best_top_k_indices = []
    best_found_at_idx = -1
    if (
        type(noise) == list
        and type(tau) == list
        and type(sk_iters) == list
        and type(opt_iters) == list
    ):
        iterable = zip(noise, tau, sk_iters, opt_iters)
    else:
        iterable = zip([noise], [tau], [sk_iters], [opt_iters])
    for noise, tau, sk_iters, opt_iters in iterable:
        for train_idx in range(opt_iters):
            gumbel_weights_float = torch.sigmoid(latent_vars)
            # noise = 1 - 0.75 * train_idx / 1000
            top_k_indices, probs = gumbel_sinkhorn_topk(
                gumbel_weights_float,
                max_covering_items,
                max_iter=sk_iters,
                tau=tau,
                sample_num=sample_num,
                noise_fact=noise,
                return_prob=True,
            )
            obj, bipartite_adj = compute_obj_differentiable(
                weights, sets, probs, bipartite_adj, probs.device
            )
            (-obj).mean().backward()
            if train_idx % 10 == 0 and verbose:
                print(
                    f"idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}"
                )
            if sample_num2 is not None and noise2 is not None:
                top_k_indices, probs = gumbel_sinkhorn_topk(
                    gumbel_weights_float,
                    max_covering_items,
                    max_iter=sk_iters,
                    tau=tau,
                    sample_num=sample_num2,
                    noise_fact=noise2,
                    return_prob=True,
                )
            obj = compute_objective(
                weights, sets, top_k_indices, bipartite_adj, device=probs.device
            )
            best_idx = torch.argmax(obj)
            max_obj, top_k_indices = obj[best_idx], top_k_indices[best_idx]
            if max_obj > best_obj:
                best_obj = max_obj
                best_top_k_indices = top_k_indices
                best_found_at_idx = train_idx
            if train_idx % 10 == 0 and verbose:
                print(
                    f"idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}"
                )
            optimizer.step()
            optimizer.zero_grad()
    return best_obj, best_top_k_indices


def lml_max_covering(weights, sets, max_covering_items, model, opt_iters, verbose=True):
    graph = build_graph_from_weights_sets(weights, sets, device_)
    latent_vars = model(graph).detach()
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=0.1)
    bipartite_adj = None
    best_obj = 0
    best_top_k_indices = []
    best_found_at_idx = -1

    for train_idx in range(opt_iters):
        weights_float = torch.sigmoid(latent_vars)
        probs = LML(N=max_covering_items)(weights_float)
        top_k_indices = torch.topk(probs, max_covering_items, dim=-1).indices
        obj, bipartite_adj = compute_obj_differentiable(
            weights, sets, probs, bipartite_adj, probs.device
        )
        (-obj).mean().backward()
        if train_idx % 10 == 0 and verbose:
            print(
                f"idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}"
            )
        obj = compute_objective(
            weights, sets, top_k_indices, bipartite_adj, device=probs.device
        )
        if obj > best_obj:
            best_obj = obj
            best_top_k_indices = top_k_indices
            best_found_at_idx = train_idx
        if train_idx % 10 == 0 and verbose:
            print(
                f"idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}"
            )
        optimizer.step()
        optimizer.zero_grad()
    return best_obj, best_top_k_indices


def gumbel_max_covering(
    weights, sets, max_covering_items, model, sample_num, noise, opt_iters, verbose=True
):
    graph = build_graph_from_weights_sets(weights, sets, device_)
    latent_vars = model(graph).detach()
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=0.001)
    bipartite_adj = None
    best_obj = 0
    best_top_k_indices = []
    best_found_at_idx = -1

    @perturbations.perturbed(
        num_samples=sample_num,
        noise="gumbel",
        sigma=noise,
        batched=False,
        device=device_,
    )
    def perturb_topk(covered_weights):
        probs = torch.zeros_like(covered_weights)
        probs[
            torch.arange(sample_num).repeat_interleave(max_covering_items),
            torch.topk(covered_weights, max_covering_items, dim=-1).indices.view(-1),
        ] = 1
        return probs

    for train_idx in range(opt_iters):
        gumbel_weights_float = torch.sigmoid(latent_vars)
        # noise = 1 - 0.75 * train_idx / 1000
        probs = perturb_topk(gumbel_weights_float)
        obj, bipartite_adj = compute_obj_differentiable(
            weights, sets, probs, bipartite_adj, probs.device
        )
        (-obj).mean().backward()
        if train_idx % 10 == 0 and verbose:
            print(
                f"idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}"
            )
        top_k_indices = torch.topk(probs, max_covering_items, dim=-1).indices
        obj = compute_objective(
            weights, sets, top_k_indices, bipartite_adj, device=probs.device
        )
        best_idx = torch.argmax(obj)
        max_obj, top_k_indices = obj[best_idx], top_k_indices[best_idx]
        if max_obj > best_obj:
            best_obj = max_obj
            best_top_k_indices = top_k_indices
            best_found_at_idx = train_idx
        if train_idx % 10 == 0 and verbose:
            print(
                f"idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}"
            )
        optimizer.step()
        optimizer.zero_grad()
    return best_obj, best_top_k_indices


def blackbox_max_covering(
    weights, sets, max_covering_items, model, lambda_param, opt_iters, verbose=True
):
    graph = build_graph_from_weights_sets(weights, sets, device_)
    latent_vars = model(graph).detach()
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=0.1)
    bipartite_adj = None
    best_obj = 0
    best_top_k_indices = []
    best_found_at_idx = -1

    bb_topk = blackbox_diff.BBTopK()
    for train_idx in range(opt_iters):
        weights_float = torch.sigmoid(latent_vars)
        # noise = 1 - 0.75 * train_idx / 1000
        probs = bb_topk.apply(weights_float, max_covering_items, lambda_param)
        obj, bipartite_adj = compute_obj_differentiable(
            weights, sets, probs, bipartite_adj, probs.device
        )
        (-obj).mean().backward()
        if train_idx % 100 == 0 and verbose:
            print(
                f"idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}"
            )
        top_k_indices = torch.topk(probs, max_covering_items, dim=-1).indices
        obj = compute_objective(
            weights, sets, top_k_indices, bipartite_adj, device=probs.device
        )
        if obj > best_obj:
            best_obj = obj
            best_top_k_indices = top_k_indices
            best_found_at_idx = train_idx
        if train_idx % 100 == 0 and verbose:
            print(
                f"idx:{train_idx} {obj.max():.1f}, {obj.mean():.1f}, best {best_obj:.0f} found at {best_found_at_idx}"
            )
        optimizer.step()
        optimizer.zero_grad()
    return best_obj, best_top_k_indices


#################################################
#        Traditional Max-kVC Methods            #
#################################################


def greedy_max_covering(weights, sets, max_selected):
    sets = deepcopy(sets)
    covered_items = set()
    selected_sets = []
    for i in range(max_selected):
        max_weight_index = -1
        max_weight = -1

        # compute the covered weights for each set
        covered_weights = []
        for current_set_index, cur_set in enumerate(sets):
            current_weight = 0
            for item in cur_set:
                if item not in covered_items:
                    current_weight += weights[item]
            covered_weights.append(current_weight)
            if current_weight > max_weight:
                max_weight = current_weight
                max_weight_index = current_set_index

        assert max_weight_index != -1
        assert max_weight >= 0

        # update the coverage status
        covered_items.update(sets[max_weight_index])
        sets[max_weight_index] = []
        selected_sets.append(max_weight_index)

    objective_score = sum([weights[item] for item in covered_items])

    return objective_score, selected_sets


def ortools_max_covering(
    weights,
    sets,
    max_selected,
    solver_name=None,
    linear_relaxation=True,
    timeout_sec=60,
):
    # define solver instance
    if solver_name is None:
        if linear_relaxation:
            solver = pywraplp.Solver(
                "DAG_scheduling", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING
            )
        else:
            solver = pywraplp.Solver(
                "DAG_scheduling", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
            )
    else:
        solver = pywraplp.Solver.CreateSolver(solver_name)

    # Initialize variables
    VarX = {}
    VarY = {}
    ConstY = {}
    for item_id, weight in enumerate(weights):
        if linear_relaxation:
            VarY[item_id] = solver.NumVar(0.0, 1.0, f"y_{item_id}")
        else:
            VarY[item_id] = solver.BoolVar(f"y_{item_id}")

    for item_id in range(len(weights)):
        ConstY[item_id] = 0

    for set_id, set_items in enumerate(sets):
        if linear_relaxation:
            VarX[set_id] = solver.NumVar(0.0, 1.0, f"x_{set_id}")
        else:
            VarX[set_id] = solver.BoolVar(f"x_{set_id}")

        # add constraint to Y
        for item_id in set_items:
            # if item_id not in ConstY:
            #    ConstY[item_id] = VarX[set_id]
            # else:
            ConstY[item_id] += VarX[set_id]

    for item_id in range(len(weights)):
        solver.Add(VarY[item_id] <= ConstY[item_id])

    # add constraint to X
    X_constraint = 0
    for set_id in range(len(sets)):
        X_constraint += VarX[set_id]
    solver.Add(X_constraint <= max_selected)

    # the objective
    Covered = 0
    for item_id in range(len(weights)):
        Covered += VarY[item_id] * weights[item_id]

    solver.Maximize(Covered)

    if timeout_sec > 0:
        solver.set_time_limit(int(timeout_sec * 1000))
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        return solver.Objective().Value(), [
            VarX[_].solution_value() for _ in range(len(sets))
        ]
    else:
        print("Did not find the optimal solution. status={}.".format(status))
        return solver.Objective().Value(), [
            VarX[_].solution_value() for _ in range(len(sets))
        ]


def gurobi_max_covering(
    weights,
    sets,
    max_selected,
    linear_relaxation=True,
    timeout_sec=60,
    start=None,
    verbose=True,
    seed=42,
):
    import gurobipy as grb

    try:
        if type(weights) is torch.Tensor:
            tensor_input = True
            device = device_
            weights = weights.cpu().numpy()
        else:
            tensor_input = False
        if start is not None and type(start) is torch.Tensor:
            start = start.cpu().numpy()

        model = grb.Model("max covering")        
        if verbose:
            model.setParam("LogToConsole", 1)
        else:
            model.setParam("LogToConsole", 0)
        if timeout_sec > 0:
            model.setParam("TimeLimit", timeout_sec)
        model.setParam("Seed", seed)

        # Initialize variables
        VarX = {}
        VarY = {}
        ConstY = {}
        for item_id, weight in enumerate(weights):
            if linear_relaxation:
                VarY[item_id] = model.addVar(
                    0.0, 1.0, vtype=grb.GRB.CONTINUOUS, name=f"y_{item_id}"
                )
            else:
                VarY[item_id] = model.addVar(vtype=grb.GRB.BINARY, name=f"y_{item_id}")
        for item_id in range(len(weights)):
            ConstY[item_id] = 0
        for set_id, set_items in enumerate(sets):
            if linear_relaxation:
                VarX[set_id] = model.addVar(
                    0.0, 1.0, vtype=grb.GRB.CONTINUOUS, name=f"x_{set_id}"
                )
            else:
                VarX[set_id] = model.addVar(vtype=grb.GRB.BINARY, name=f"x_{set_id}")
            if start is not None:
                VarX[set_id].start = start[set_id]

            # add constraint to Y
            for item_id in set_items:
                ConstY[item_id] += VarX[set_id]
        for item_id in range(len(weights)):
            model.addConstr(VarY[item_id] <= ConstY[item_id])

        # add constraint to X
        X_constraint = 0
        for set_id in range(len(sets)):
            X_constraint += VarX[set_id]
        model.addConstr(X_constraint <= max_selected)

        # the objective
        Covered = 0
        for item_id in range(len(weights)):
            Covered += VarY[item_id] * weights[item_id]
        model.setObjective(Covered, grb.GRB.MAXIMIZE)

        model.optimize()

        res = [model.getVarByName(f"x_{set_id}").X for set_id in range(len(sets))]
        if tensor_input:
            res = np.array(res, dtype=np.int)
            return model.getObjective().getValue(), torch.from_numpy(res).to(device)
        else:
            return model.getObjective().getValue(), res

    except grb.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))

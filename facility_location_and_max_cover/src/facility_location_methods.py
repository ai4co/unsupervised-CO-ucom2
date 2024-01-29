import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple
from ortools.linear_solver import pywraplp
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
import time

from tqdm import trange
from src.gumbel_sinkhorn_topk import gumbel_sinkhorn_topk
from src.poi_bin import pmf_poibin, pmf_poibin_vec
from torch_geometric.utils import dropout_adj

####################################
#        helper functions          #
####################################

device = torch.device("cuda:0")


def compute_objective(
    points, cluster_centers, distance_metric="euclidean", choice_cluster=None
):
    dist_func = _get_distance_func(distance_metric)
    dist = dist_func(points, cluster_centers, points.device)
    if choice_cluster is None:
        choice_cluster = torch.argmin(dist, dim=-1)
    return torch.sum(
        torch.gather(dist, -1, choice_cluster.unsqueeze(-1)).squeeze(-1), dim=-1
    )


# Want et al. ICLR'23
def compute_objective_differentiable(dist, probs, temp=30):
    exp_dist = torch.exp(-temp / dist.mean() * dist)
    exp_dist_probs = exp_dist.unsqueeze(0) * probs.unsqueeze(-1)
    probs_per_dist = exp_dist_probs / (exp_dist_probs.sum(1, keepdim=True))
    obj = (probs_per_dist * dist).sum(dim=(1, 2))
    return obj


# UCom2 derivation
def compute_objective_differentiable_exact(dist, probs):
    probs = probs.flatten()
    dist_sort = torch.sort(dist, dim=1)
    dist_ordering = dist_sort.indices
    dist_ordered = dist_sort.values
    p_reordered = probs[dist_ordering]
    q_reordered = 1 - p_reordered
    q_reordered_cumprod = q_reordered.cumprod(dim=1).roll(shifts=1, dims=-1)
    q_reordered_cumprod[:, 0] = 1.0
    p_closest = q_reordered_cumprod * p_reordered
    obj = (dist_ordered * p_closest).sum()
    return obj


def build_graph_from_points(
    points, dist=None, return_dist=False, distance_metric="euclidean"
):
    if dist is None:
        dist_func = _get_distance_func(distance_metric)
        dist = dist_func(points, points, points.device)
    norm_dist = dist * 1.414 / dist.max()
    edge_indices = torch.nonzero(norm_dist <= 0.02, as_tuple=False).transpose(0, 1)
    edge_attrs = (points.unsqueeze(0) - points.unsqueeze(1))[
        torch.nonzero(norm_dist <= 0.02, as_tuple=True)
    ] + 0.5
    g = pyg.data.Data(x=points, edge_index=edge_indices, edge_attr=edge_attrs)
    if return_dist:
        return g, dist
    else:
        return g


#################################################
#             Learning FLP Methods              #
#################################################


class GNNModel(torch.nn.Module):
    # clustering model (3-layer SplineConv)
    def __init__(self):
        super(GNNModel, self).__init__()
        self.gconv1 = pyg.nn.SplineConv(2, 16, 2, 5)
        self.gconv2 = pyg.nn.SplineConv(16, 16, 2, 5)
        self.gconv3 = pyg.nn.SplineConv(16, 16, 2, 5)
        self.fc = torch.nn.Linear(16, 1)

    def forward(self, g, return_log=False):
        x = torch.relu(self.gconv1(g.x, g.edge_index, g.edge_attr))
        x = torch.relu(self.gconv2(x, g.edge_index, g.edge_attr))
        x = torch.relu(self.gconv3(x, g.edge_index, g.edge_attr))
        x = self.fc(x).squeeze(-1)
        if return_log:
            return torch.sigmoid(x), x
        return torch.sigmoid(x)

    def zero_params(self):
        for param in self.parameters():
            param.zero_()


# Main obj by Want et al. ICLR'23
def egn_facility_location(
    points,
    num_clusters,
    model,
    softmax_temp,
    egn_beta,
    random_trials=0,
    time_limit=-1,
    distance_metric="euclidean",
):
    prev_time = time.time()
    graph, dist = build_graph_from_points(points, None, True, distance_metric)
    graph.ori_x = graph.x.clone()
    best_objective = float("inf")
    best_selected_indices = None
    for i_trail in range(random_trials if random_trials > 0 else 1):
        if time_limit > 0 and time.time() - prev_time > time_limit:
            break
        if random_trials > 0 and i_trail > 0:
            graph.x = graph.ori_x + torch.randn_like(graph.x) / 100
        probs = model(graph).detach()
        dist_probs, probs_argsort = torch.sort(probs, descending=True)
        selected_items = 0
        for prob_idx in probs_argsort:
            if selected_items >= num_clusters:
                probs[prob_idx] = 0
                continue
            probs_0 = probs.clone()
            probs_0[prob_idx] = 0
            probs_1 = probs.clone()
            probs_1[prob_idx] = 1
            constraint_conflict_0 = torch.relu(probs_0.sum() - num_clusters)
            constraint_conflict_1 = torch.relu(probs_1.sum() - num_clusters)
            obj_0 = (
                compute_objective_differentiable(dist, probs_0, temp=softmax_temp)
                + egn_beta * constraint_conflict_0
            )
            obj_1 = (
                compute_objective_differentiable(dist, probs_1, temp=softmax_temp)
                + egn_beta * constraint_conflict_1
            )
            if obj_0 >= obj_1:
                probs[prob_idx] = 1
                selected_items += 1
            else:
                probs[prob_idx] = 0
        top_k_indices = torch.topk(probs, num_clusters, dim=-1).indices
        cluster_centers = torch.stack(
            [
                torch.gather(points[:, _], 0, top_k_indices)
                for _ in range(points.shape[1])
            ],
            dim=-1,
        )
        choice_cluster, cluster_centers, selected_indices = discrete_kmeans(
            points,
            num_clusters,
            init_x=cluster_centers,
            distance=distance_metric,
            device=points.device,
        )
        objective = compute_objective(points, cluster_centers, distance_metric).item()
        if objective < best_objective:
            best_objective = objective
            best_selected_indices = selected_indices
    return best_objective, best_selected_indices, time.time() - prev_time


# Main obj from Ucom2
def egn_facility_location_exact(
    points,
    num_clusters,
    model,
    # softmax_temp,
    egn_beta,
    random_trials=0,
    time_limit=-1,
    distance_metric="euclidean",
):
    prev_time = time.time()
    graph, dist = build_graph_from_points(points, None, True, distance_metric)
    graph.ori_x = graph.x.clone()
    best_objective = float("inf")
    best_selected_indices = None
    for i_trail in range(random_trials if random_trials > 0 else 1):
        if time_limit > 0 and time.time() - prev_time > time_limit:
            break
        if random_trials > 0 and i_trail > 0:
            graph_copy = graph.clone()
            graph_copy.x = graph_copy.x + torch.randn_like(graph.x) * 0.2
            graph_copy.edge_index, graph_copy.edge_attr = dropout_adj(
                graph.edge_index, graph.edge_attr, p=0.2
            )
        else:
            graph_copy = graph.clone()
        probs = model(graph_copy).detach()
        # print(probs.sum())
        dist_probs, probs_argsort = torch.sort(probs, descending=True)
        selected_items = 0
        for prob_idx in probs_argsort:
            if selected_items >= num_clusters:
                probs[prob_idx] = 0
                continue
            probs_0 = probs.clone()
            probs_0[prob_idx] = 0
            probs_1 = probs.clone()
            probs_1[prob_idx] = 1
            constraint_conflict_0 = torch.relu(probs_0.sum() - num_clusters)
            constraint_conflict_1 = torch.relu(probs_1.sum() - num_clusters)
            # constraint_conflict_0 = torch.abs(probs_0.sum() - num_clusters)
            # constraint_conflict_1 = torch.abs(probs_1.sum() - num_clusters)
            # obj_0 = compute_objective_differentiable(dist, probs_0,
            #                                          temp=softmax_temp) + egn_beta * constraint_conflict_0
            # obj_1 = compute_objective_differentiable(dist, probs_1,
            #                                          temp=softmax_temp) + egn_beta * constraint_conflict_1
            obj_0 = (
                compute_objective_differentiable_exact(dist, probs_0)
                + egn_beta * constraint_conflict_0
            )
            obj_1 = (
                compute_objective_differentiable_exact(dist, probs_1)
                + egn_beta * constraint_conflict_1
            )
            if obj_0 >= obj_1:
                probs[prob_idx] = 1
                selected_items += 1
            else:
                probs[prob_idx] = 0
        top_k_indices = torch.topk(probs, num_clusters, dim=-1).indices
        cluster_centers = torch.stack(
            [
                torch.gather(points[:, _], 0, top_k_indices)
                for _ in range(points.shape[1])
            ],
            dim=-1,
        )
        choice_cluster, cluster_centers, selected_indices = discrete_kmeans(
            points,
            num_clusters,
            init_x=cluster_centers,
            distance=distance_metric,
            device=points.device,
        )
        objective = compute_objective(points, cluster_centers, distance_metric).item()
        if objective < best_objective:
            best_objective = objective
            best_selected_indices = selected_indices
    return best_objective, best_selected_indices, time.time() - prev_time


def egn_greedy_facility_location_exact(
    points,
    num_clusters,
    model,
    softmax_temp,
    egn_beta,
    random_trials=0,
    time_limit=-1,
    distance_metric="euclidean",
):
    prev_time = time.time()
    graph, dist = build_graph_from_points(points, None, True, distance_metric)
    graph.ori_x = graph.x.clone()
    best_objective = float("inf")
    best_selected_indices = None
    for i_trail in range(random_trials if random_trials > 0 else 1):
        if time_limit > 0 and time.time() - prev_time > time_limit:
            break
        if random_trials > 0 and i_trail > 0:
            graph.x = graph.ori_x + torch.randn_like(graph.x) / 100
        probs = model(graph).detach()

        improved = True
        improved_times = 0

        while improved:
            improved = False
            vx2result = torch.empty(probs.shape[0], 2).to(probs.device)
            for i in range(probs.shape[0]):
                probs_0 = probs.clone()
                probs_0[i] = 0
                obj_0 = compute_objective_differentiable_exact(dist, probs_0)
                constraint_conflict_0 = torch.relu(probs_0.sum() - num_clusters)
                vx2result[i, 0] = obj_0 + egn_beta * constraint_conflict_0

                probs_0[i] = 1
                obj_1 = compute_objective_differentiable_exact(dist, probs_0)
                constraint_conflict_1 = torch.relu(probs_0.sum() - num_clusters)
                vx2result[i, 1] = obj_1 + egn_beta * constraint_conflict_1

            best_index = torch.argmin(vx2result)
            best_v, best_x = best_index // 2, best_index % 2
            if probs[best_v] != best_x:
                probs[best_v] = best_x
                improved = True
                improved_times += 1
                print(improved_times, end="\r")
        top_k_indices = torch.topk(probs, num_clusters, dim=-1).indices.flatten()
        cluster_centers = torch.stack(
            [
                torch.gather(points[:, _], 0, top_k_indices)
                for _ in range(points.shape[1])
            ],
            dim=-1,
        )
        choice_cluster, cluster_centers, selected_indices = discrete_kmeans(
            points,
            num_clusters,
            init_x=cluster_centers,
            distance=distance_metric,
            device=points.device,
        )
        objective = compute_objective(points, cluster_centers, distance_metric).item()
        if objective < best_objective:
            best_objective = objective
            best_selected_indices = selected_indices
    return best_objective, best_selected_indices, time.time() - prev_time


def egn_pb_simple_facility_location(
    points,
    num_clusters,
    model,
    softmax_temp,
    egn_beta,
    random_trials=0,
    time_limit=-1,
    distance_metric="euclidean",
):
    prev_time = time.time()
    graph, dist = build_graph_from_points(points, None, True, distance_metric)
    graph.ori_x = graph.x.clone()
    best_objective = float("inf")
    best_selected_indices = None
    for i_trail in range(random_trials if random_trials > 0 else 1):
        if time_limit > 0 and time.time() - prev_time > time_limit:
            break
        if random_trials > 0 and i_trail > 0:
            graph.x = graph.ori_x + torch.randn_like(graph.x) / 100
        probs = model(graph).detach()
        # print(probs.sum())
        dist_probs, probs_argsort = torch.sort(probs, descending=True)
        selected_items = 0
        k_diff = torch.relu(
            torch.arange(probs.shape[0] + 1, device=device) - num_clusters
        )
        # k_diff = torch.abs(torch.arange(probs.shape[0] + 1, device=device) - num_clusters)

        for prob_idx in probs_argsort:
            if selected_items >= num_clusters:
                probs[prob_idx] = 0
                continue
            probs_0 = probs.clone()
            probs_0[prob_idx] = 0
            card_dist_0 = pmf_poibin_vec(
                probs_0, points.device, use_normalization=False
            )
            constraint_conflict_0 = (card_dist_0 * k_diff).sum()
            probs_1 = probs.clone()
            probs_1[prob_idx] = 1
            card_dist_1 = pmf_poibin_vec(
                probs_1, points.device, use_normalization=False
            )
            constraint_conflict_1 = (card_dist_1 * k_diff).sum()
            # constraint_conflict_0 = torch.relu(probs_0.sum() - num_clusters)
            # constraint_conflict_1 = torch.relu(probs_1.sum() - num_clusters)
            obj_0 = (
                compute_objective_differentiable(dist, probs_0, temp=softmax_temp)
                + egn_beta * constraint_conflict_0
            )
            obj_1 = (
                compute_objective_differentiable(dist, probs_1, temp=softmax_temp)
                + egn_beta * constraint_conflict_1
            )
            if obj_0 >= obj_1:
                probs[prob_idx] = 1
                selected_items += 1
            else:
                probs[prob_idx] = 0
        top_k_indices = torch.topk(probs, num_clusters, dim=-1).indices
        cluster_centers = torch.stack(
            [
                torch.gather(points[:, _], 0, top_k_indices)
                for _ in range(points.shape[1])
            ],
            dim=-1,
        )
        choice_cluster, cluster_centers, selected_indices = discrete_kmeans(
            points,
            num_clusters,
            init_x=cluster_centers,
            distance=distance_metric,
            device=points.device,
        )
        objective = compute_objective(points, cluster_centers, distance_metric).item()
        if objective < best_objective:
            best_objective = objective
            best_selected_indices = selected_indices
    return best_objective, best_selected_indices, time.time() - prev_time


@torch.no_grad()
def egn_pb_greedy_facility_location_exact_fast_test_truncated_soft_then_hard(
    points,
    graph,
    dist,
    num_clusters,
    model,
    egn_beta,
    random_trials=0,
    num_rounds=1,
    time_limit=-1,
    noise_scale=0.1,
    stru_noise=0.05,
    # test_time_opt=False,
    # inner_lr=0.1,
    # inner_ep=100,
    # tto_sigmoid=False,
    distance_metric="euclidean",
    s_max=-1,
    hard_after=True,
):
    prev_time = time.time()
    best_objective = float("inf")
    best_selected_indices = None

    best_objective_list = []
    for i_round in range(num_rounds):
        if i_round == 0:
            graph_list = [graph]
            for i_trail in range(random_trials - 1):
                graph_copy = graph.clone()
                graph_copy.x = graph_copy.x + torch.randn_like(graph.x) * noise_scale
                if stru_noise > 0:
                    graph_copy.edge_index, graph_copy.edge_attr = dropout_adj(
                        graph.edge_index, graph.edge_attr, p=stru_noise
                    )
                graph_list.append(graph_copy)
        else:
            graph_list = []
            for i_trail in range(random_trials):
                graph_copy = graph.clone()
                graph_copy.x = graph_copy.x + torch.randn_like(graph.x) * noise_scale
                if stru_noise > 0:
                    graph_copy.edge_index, graph_copy.edge_attr = dropout_adj(
                        graph.edge_index, graph.edge_attr, p=stru_noise
                    )
                graph_list.append(graph_copy)
        loader = DataLoader(graph_list, batch_size=len(graph_list))
        graph_batch = next(iter(loader))
        probs_batch = model(graph_batch, return_log=False)
        n_nodes = graph.x.shape[0]
        probs_matrix = probs_batch.view(-1, n_nodes)
        k = num_clusters
        if s_max <= 0:
            s_max = n_nodes + 1
        k_diff_remove = torch.abs(
            torch.arange(n_nodes, device=probs_matrix.device) - k
        )  # [n]
        k_diff_remove[:2] *= 1000
        k_diff_add = torch.abs(
            torch.arange(n_nodes, device=probs_matrix.device) + 1 - k
        )  # [n]
        k_diff_add[:1] *= 1000
        # k_diff = torch.abs(torch.arange(probs_matrix.shape[-1] + 1, device=device) - k)
        for i_soft_hard in range(2 if hard_after else 1):
            if i_soft_hard == 1:
                egn_beta = float(n_nodes)
            improved = [True for _ in range(random_trials)]
            while any(improved):
                if time_limit > 0 and time.time() - prev_time > time_limit:
                    break
                card_dist_orig = pmf_poibin(
                    probs_matrix, probs_matrix.device, use_normalization=False
                )
                dist_sort = torch.sort(dist, dim=1)
                dist_ordering = dist_sort.indices
                dist_ordered = dist_sort.values

                # p_reordered = [b, n, n]; p_reodered[i, j, k] = probs_matrix[i, dist_ordering[j, k]]
                p_reordered = probs_matrix[:, dist_ordering]
                q_reordered = 1 - p_reordered

                q_reordered_cumprod = q_reordered.cumprod(dim=-1).roll(
                    shifts=1, dims=-1
                )
                q_reordered_cumprod[..., 0] = 1.0
                p_closest = q_reordered_cumprod * p_reordered

                # p2, constriant 2
                xx_ = probs_matrix
                pmf_cur = card_dist_orig
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
                    # )[:, :n_nodes]
                )[:, :n_nodes, :s_max]
                # q_roll_stack[ib, i] = (q_i, q_{i-1}, ..., q_0, 0, ..., 0), 0 <= i < n; [b, n, n + 1]
                term_2a = (xx_ / (xx_ - 1.0)).unsqueeze(-1) ** torch.arange(
                    # n_nodes + 1, device=probs_matrix.device
                    s_max,
                    device=probs_matrix.device,
                )
                # term_2a[ib, iv, i] = (X^{(b)}_v / (X^{(b)}_v - 1))^{i}; [b, n, n + 1]
                # term_2a = torch.einsum("bix, bvx -> bvi", q_roll_stack, term_2a)  # [b, n, n]
                term_2a = term_2a @ q_roll_stack.transpose(1, 2)  # [b, n, n]

                res_case_1 = term_1a * term_2a
                # del term_1a, term_2a

                # [0.5, 1]
                term_1b = (1.0 / xx_).unsqueeze(-1)
                q = pmf_cur
                q_roll_stack = torch.tril(
                    torch.as_strided(
                        q.repeat(1, 2),
                        (q.shape[0], q.shape[1], q.shape[1]),
                        (q.shape[1] * 2, 1, 1),
                    ).flip(1)
                    # )[:, :n_nodes].flip(1)
                )[:, :n_nodes].flip(1)[..., :s_max]
                # q_roll_stack[ib, i] = (q_{i + 1}, q_{i + 2}, ..., q_n, 0, ..., 0), 0 <= i < n; [b, n, n + 1]
                term_2b = ((xx_ - 1.0) / xx_).unsqueeze(-1) ** torch.arange(
                    # n_nodes + 1, device=probs_matrix.device
                    s_max,
                    device=probs_matrix.device,
                )
                # term_2b = torch.einsum("bix, bvx -> bvi", q_roll_stack, term_2b)
                term_2b = term_2b @ q_roll_stack.transpose(1, 2)  # [b, n, n]

                res_case_2 = term_1b * term_2b
                # del term_1b, term_2b

                tilde_q = torch.where(xx_.unsqueeze(-1) <= 0.5, res_case_1, res_case_2)
                tilde_q.clamp_(0.0, 1.0)

                pmf_new = tilde_q[..., :n_nodes]  # [b, n, n]
                dol_remove_p2 = (pmf_new * k_diff_remove).sum(dim=-1)  # [b, n]
                dol_add_p2 = (pmf_new * k_diff_add).sum(dim=-1)  # [b, n]

                vx2result = torch.empty(
                    probs_matrix.shape[0], probs_matrix.shape[1], 2
                ).to(probs_matrix.device)
                vx2result[..., 0] = dol_remove_p2 * egn_beta
                vx2result[..., 1] = dol_add_p2 * egn_beta

                obj_expand = dist_ordered * p_closest  # [b, n, n]; (b, v, i) -> p_i d_i
                p_closest_cumsum = (
                    p_closest.flip(-1).cumsum(-1).flip(-1).roll(-1, dims=-1)
                )
                p_closest_cumsum[..., -1] = 0  # [n, n]; (v, i) -> \sum_{j > i} p_j
                pd_cumsum = obj_expand.flip(-1).cumsum(-1).flip(-1).roll(-1, dims=-1)
                pd_cumsum[..., -1] = 0  # [n, n]; (v, i) -> \sum_{j > i} p_j d_j
                # p_ratio = p_reordered / (1 - p_reordered)  # [n, n]; (v, i) -> X_{ui} / (1 - X_{ui})
                p_ratio = probs_matrix / (
                    1 - probs_matrix
                )  # [n]; (u) -> X_u / (1 - X_u)
                inv_ordering = dist_ordering.argsort(dim=-1).expand(
                    random_trials, -1, -1
                )
                # fv_diff_add(u) = \sum_v p_closest_cumsum[v, inv_ordering[v, u]] dist[v, u] - pd_cumsum[v, inv_ordering[v, u]]
                fv_diff_add = (
                    torch.gather(p_closest_cumsum, dim=-1, index=inv_ordering) * dist
                    - torch.gather(pd_cumsum, dim=-1, index=inv_ordering)
                ).sum(dim=-2)
                # fv_diff_remove(u) = \sum_v pd_cumsum[v, inv_ordering[v, u]] * p_ratio[v, inv_ordering[v, u]] - p_cloest[v, inv_ordering[v, u]] * dist[v, u]
                # fv_diff_remove(u) = \sum_v pd_cumsum[v, inv_ordering[v, u]] * p_ratio[u] - p_cloest[v, inv_ordering[v, u]] * dist[v, u]
                fv_diff_remove = (
                    torch.gather(pd_cumsum, dim=-1, index=inv_ordering)
                    * p_ratio.unsqueeze(-2)
                    - torch.gather(p_closest, dim=-1, index=inv_ordering) * dist
                ).sum(dim=-2)
                vx2result[..., 0] += fv_diff_remove
                vx2result[..., 1] += fv_diff_add

                vx2result_flatten = vx2result.view(random_trials, -1)
                k_top = 1
                vx2result_flatten_topk = torch.topk(
                    vx2result_flatten, k_top, largest=False
                ).indices

                for i_input in range(random_trials):
                    if not improved[i_input]:
                        continue
                    improved[i_input] = False
                    for best_index in vx2result_flatten_topk[i_input]:
                        if improved[i_input]:
                            break
                        best_v, best_x = best_index // 2, best_index % 2
                        if torch.abs(probs_matrix[i_input, best_v] - best_x) >= 0.01:
                            probs_matrix[i_input, best_v] = 0.001 + 0.998 * best_x
                            improved[i_input] = True
            top_k_indices = torch.topk(probs_matrix, k, dim=-1).indices
            cluster_centers = points[top_k_indices]
            # cluster_centers = torch.stack([torch.gather(points[:, _], 0, top_k_indices) for _ in range(points.shape[1])],
            #                               dim=-1)
            # objective = compute_objective(points, cluster_centers, distance_metric)
            # best_idx = torch.argmin(objective)
            # top_k_indices = top_k_indices[best_idx]
            # cluster_centers = points[top_k_indices]
        for top_k_ind in top_k_indices:
            cluster_centers = points[top_k_ind]
            choice_cluster, cluster_centers, selected_indices = discrete_kmeans(
                points,
                num_clusters,
                init_x=cluster_centers,
                distance=distance_metric,
                device=points.device,
            )
            # except:
            #     breakpoint()
            objective = compute_objective(
                points, cluster_centers, distance_metric
            ).item()
            if objective < best_objective:
                best_objective = objective
                best_selected_indices = selected_indices
            best_objective_list.append(best_objective)
    return (
        best_objective,
        best_selected_indices,
        time.time() - prev_time,
        best_objective_list,
    )


def sinkhorn_facility_location(
    points,
    num_clusters,
    model,
    softmax_temp,
    sample_num,
    noise,
    tau,
    sk_iters,
    opt_iters,
    sample_num2=None,
    noise2=None,
    grad_step=0.1,
    time_limit=-1,
    distance_metric="euclidean",
    verbose=True,
    exact_obj=False,
    iter_ratio=1.0,
):
    prev_time = time.time()
    # dist_sorted, dist_argsort = torch.sort(dist, dim=1)
    graph, dist = build_graph_from_points(points, None, True, distance_metric)
    # latent_vars = torch.rand(points.shape[0], device=points.device, requires_grad=True)
    latent_vars = model(graph).detach()
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=grad_step)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5000], 0.1)
    best_obj = float("inf")
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
    opt_iter_offset = 0
    for noise, tau, sk_iters, opt_iters in iterable:
        opt_iters = int(opt_iters * iter_ratio)
        for opt_idx in range(opt_iter_offset, opt_iter_offset + opt_iters):
            # time limit control
            if time_limit > 0 and time.time() - prev_time > time_limit:
                break

            gumbel_weights_float = torch.sigmoid(latent_vars)
            top_k_indices, probs = gumbel_sinkhorn_topk(
                gumbel_weights_float,
                num_clusters,
                max_iter=sk_iters,
                tau=tau,
                sample_num=sample_num,
                noise_fact=noise,
                return_prob=True,
            )
            # compute objective by softmax
            if exact_obj:
                obj = compute_objective_differentiable_exact(dist, probs)
            else:
                obj = compute_objective_differentiable(dist, probs, temp=softmax_temp)

            """
            # compute objective by argmin
            sorted_probs = probs[:, dist_argsort]
            cumsum_sorted_probs = torch.cumsum(sorted_probs, dim=2)
            mask = (cumsum_sorted_probs <= 1).to(dtype=torch.float)
            def index_3dtensor_by_2dmask(t, m):
                return torch.gather(t, 2, m.sum(dim=-1, keepdim=True).to(dtype=torch.long))
            t = - (index_3dtensor_by_2dmask(cumsum_sorted_probs, mask) - 1) #/ index_3dtensor_by_2dmask(sorted_probs, mask)
            new_probs = sorted_probs.scatter_add(2, mask.sum(dim=-1, keepdim=True).to(dtype=torch.long), t)
            #t = 1 / index_3dtensor_by_2dmask(cumsum_sorted_probs, mask)
            #new_probs = sorted_probs / index_3dtensor_by_2dmask(cumsum_sorted_probs, mask)
            new_mask = mask.scatter(2, mask.sum(dim=-1).to(dtype=torch.long).unsqueeze(-1),
                                    torch.ones(mask.shape[0], mask.shape[1], 1, device=points.device))
            probs_with_dist = new_mask * new_probs * dist_sorted.unsqueeze(0)
            obj = probs_with_dist.sum(dim=(1, 2))
            """
            obj.mean().backward()
            if opt_idx % 10 == 0 and verbose:
                print(
                    f"idx:{opt_idx} estimated {obj.min():.4f}, {obj.mean():.4f}, best {best_obj:.4f} found at {best_found_at_idx}"
                )
            if sample_num2 is not None and noise2 is not None:
                top_k_indices, probs = gumbel_sinkhorn_topk(
                    gumbel_weights_float,
                    num_clusters,
                    max_iter=100,
                    tau=0.05,
                    sample_num=sample_num2,
                    noise_fact=noise2,
                    return_prob=True,
                )
            cluster_centers = torch.gather(
                torch.repeat_interleave(points.unsqueeze(0), top_k_indices.shape[0], 0),
                1,
                torch.repeat_interleave(
                    top_k_indices.unsqueeze(-1), points.shape[-1], -1
                ),
            )
            obj = compute_objective(
                points.unsqueeze(0), cluster_centers, distance_metric
            )
            best_idx = torch.argmin(obj)
            min_obj, top_k_indices = obj[best_idx], top_k_indices[best_idx]
            # choice_cluster, cluster_centers, selected_indices = discrete_kmeans(
            #    points, num_clusters, init_x=cluster_centers[best_idx], distance=distance_metric, device=points.device)
            # min_obj = compute_objective(points, cluster_centers, distance_metric).item()
            if min_obj < best_obj:
                best_obj = min_obj
                best_top_k_indices = top_k_indices
                best_found_at_idx = opt_idx
            if opt_idx % 10 == 0 and verbose:
                print(
                    f"idx:{opt_idx} real {obj.min():.4f}, {obj.mean():.4f}, best {best_obj:.4f} found at {best_found_at_idx}, now time:{time.time()-prev_time:.2f}"
                )
            optimizer.step()
            optimizer.zero_grad()
            # lr_scheduler.step()
        opt_iter_offset += opt_iters
    cluster_centers = torch.stack(
        [
            torch.gather(points[:, _], 0, best_top_k_indices)
            for _ in range(points.shape[1])
        ],
        dim=-1,
    )
    objective = compute_objective(points, cluster_centers, distance_metric).item()
    # print(f'{index} gumbel objective={objective:.4f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - prev_time}')
    # print(cluster_centers)
    # plt.plot(cluster_centers[:, 0].cpu(), cluster_centers[:, 1].cpu(), 'r+', label='gumbel')

    choice_cluster, cluster_centers, selected_indices = discrete_kmeans(
        points,
        num_clusters,
        init_x=cluster_centers,
        distance=distance_metric,
        device=points.device,
    )
    objective = compute_objective(points, cluster_centers, distance_metric).item()
    # plt.plot(cluster_centers[:, 0].cpu(), cluster_centers[:, 1].cpu(), 'kx', label='gumbel kmeans')
    return objective, selected_indices, time.time() - prev_time


def sinkhorn_facility_location_noTTO(
    points,
    num_clusters,
    model,
    softmax_temp,
    sample_num,
    noise,
    tau,
    sk_iters,
    opt_iters,
    sample_num2=None,
    noise2=None,
    grad_step=0.1,
    time_limit=-1,
    distance_metric="euclidean",
    verbose=True,
    exact_obj=False,
    iter_ratio=1.0,
):
    prev_time = time.time()
    # dist_sorted, dist_argsort = torch.sort(dist, dim=1)
    graph, dist = build_graph_from_points(points, None, True, distance_metric)
    # latent_vars = torch.rand(points.shape[0], device=points.device, requires_grad=True)
    latent_vars = model(graph).detach()
    # latent_vars.requires_grad_(True)
    # optimizer = torch.optim.Adam([latent_vars], lr=grad_step)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5000], 0.1)
    best_obj = float("inf")
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
    opt_iter_offset = 0
    for noise, tau, sk_iters, opt_iters in iterable:
        opt_iters = int(opt_iters * iter_ratio)
        for opt_idx in range(opt_iter_offset, opt_iter_offset + opt_iters):
            # time limit control
            if time_limit > 0 and time.time() - prev_time > time_limit:
                break

            gumbel_weights_float = torch.sigmoid(latent_vars)
            top_k_indices, probs = gumbel_sinkhorn_topk(
                gumbel_weights_float,
                num_clusters,
                max_iter=sk_iters,
                tau=tau,
                sample_num=sample_num,
                noise_fact=noise,
                return_prob=True,
            )
            # compute objective by softmax
            if exact_obj:
                obj = compute_objective_differentiable_exact(dist, probs)
            else:
                obj = compute_objective_differentiable(dist, probs, temp=softmax_temp)

            """
            # compute objective by argmin
            sorted_probs = probs[:, dist_argsort]
            cumsum_sorted_probs = torch.cumsum(sorted_probs, dim=2)
            mask = (cumsum_sorted_probs <= 1).to(dtype=torch.float)
            def index_3dtensor_by_2dmask(t, m):
                return torch.gather(t, 2, m.sum(dim=-1, keepdim=True).to(dtype=torch.long))
            t = - (index_3dtensor_by_2dmask(cumsum_sorted_probs, mask) - 1) #/ index_3dtensor_by_2dmask(sorted_probs, mask)
            new_probs = sorted_probs.scatter_add(2, mask.sum(dim=-1, keepdim=True).to(dtype=torch.long), t)
            #t = 1 / index_3dtensor_by_2dmask(cumsum_sorted_probs, mask)
            #new_probs = sorted_probs / index_3dtensor_by_2dmask(cumsum_sorted_probs, mask)
            new_mask = mask.scatter(2, mask.sum(dim=-1).to(dtype=torch.long).unsqueeze(-1),
                                    torch.ones(mask.shape[0], mask.shape[1], 1, device=points.device))
            probs_with_dist = new_mask * new_probs * dist_sorted.unsqueeze(0)
            obj = probs_with_dist.sum(dim=(1, 2))
            """
            # obj.mean().backward()
            if opt_idx % 10 == 0 and verbose:
                print(
                    f"idx:{opt_idx} estimated {obj.min():.4f}, {obj.mean():.4f}, best {best_obj:.4f} found at {best_found_at_idx}"
                )
            if sample_num2 is not None and noise2 is not None:
                top_k_indices, probs = gumbel_sinkhorn_topk(
                    gumbel_weights_float,
                    num_clusters,
                    max_iter=100,
                    tau=0.05,
                    sample_num=sample_num2,
                    noise_fact=noise2,
                    return_prob=True,
                )
            cluster_centers = torch.gather(
                torch.repeat_interleave(points.unsqueeze(0), top_k_indices.shape[0], 0),
                1,
                torch.repeat_interleave(
                    top_k_indices.unsqueeze(-1), points.shape[-1], -1
                ),
            )
            obj = compute_objective(
                points.unsqueeze(0), cluster_centers, distance_metric
            )
            best_idx = torch.argmin(obj)
            min_obj, top_k_indices = obj[best_idx], top_k_indices[best_idx]
            # choice_cluster, cluster_centers, selected_indices = discrete_kmeans(
            #    points, num_clusters, init_x=cluster_centers[best_idx], distance=distance_metric, device=points.device)
            # min_obj = compute_objective(points, cluster_centers, distance_metric).item()
            if min_obj < best_obj:
                best_obj = min_obj
                best_top_k_indices = top_k_indices
                best_found_at_idx = opt_idx
            if opt_idx % 10 == 0 and verbose:
                print(
                    f"idx:{opt_idx} real {obj.min():.4f}, {obj.mean():.4f}, best {best_obj:.4f} found at {best_found_at_idx}, now time:{time.time()-prev_time:.2f}"
                )
            # optimizer.step()
            # optimizer.zero_grad()
            # lr_scheduler.step()
        opt_iter_offset += opt_iters
    cluster_centers = torch.stack(
        [
            torch.gather(points[:, _], 0, best_top_k_indices)
            for _ in range(points.shape[1])
        ],
        dim=-1,
    )
    objective = compute_objective(points, cluster_centers, distance_metric).item()
    # print(f'{index} gumbel objective={objective:.4f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - prev_time}')
    # print(cluster_centers)
    # plt.plot(cluster_centers[:, 0].cpu(), cluster_centers[:, 1].cpu(), 'r+', label='gumbel')

    choice_cluster, cluster_centers, selected_indices = discrete_kmeans(
        points,
        num_clusters,
        init_x=cluster_centers,
        distance=distance_metric,
        device=points.device,
    )
    objective = compute_objective(points, cluster_centers, distance_metric).item()
    # plt.plot(cluster_centers[:, 0].cpu(), cluster_centers[:, 1].cpu(), 'kx', label='gumbel kmeans')
    return objective, selected_indices, time.time() - prev_time


# Wang et al. ICLR'23 but the main obj is from Ucom2
def sinkhorn_facility_location_exact(
    points,
    num_clusters,
    model,
    softmax_temp,
    sample_num,
    noise,
    tau,
    sk_iters,
    opt_iters,
    sample_num2=None,
    noise2=None,
    grad_step=0.1,
    time_limit=-1,
    distance_metric="euclidean",
    verbose=True,
):
    prev_time = time.time()
    # dist_sorted, dist_argsort = torch.sort(dist, dim=1)
    graph, dist = build_graph_from_points(points, None, True, distance_metric)
    # latent_vars = torch.rand(points.shape[0], device=points.device, requires_grad=True)
    latent_vars = model(graph).detach()
    latent_vars.requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vars], lr=grad_step)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5000], 0.1)
    best_obj = float("inf")
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
    opt_iter_offset = 0
    for noise, tau, sk_iters, opt_iters in iterable:
        for opt_idx in range(opt_iter_offset, opt_iter_offset + opt_iters):
            # time limit control
            if time_limit > 0 and time.time() - prev_time > time_limit:
                break

            gumbel_weights_float = torch.sigmoid(latent_vars)
            top_k_indices, probs = gumbel_sinkhorn_topk(
                gumbel_weights_float,
                num_clusters,
                max_iter=sk_iters,
                tau=tau,
                sample_num=sample_num,
                noise_fact=noise,
                return_prob=True,
            )
            # compute objective by softmax
            # obj = compute_objective_differentiable(dist, probs, temp=softmax_temp)
            obj = compute_objective_differentiable_exact(dist, probs)
            """
            # compute objective by argmin
            sorted_probs = probs[:, dist_argsort]
            cumsum_sorted_probs = torch.cumsum(sorted_probs, dim=2)
            mask = (cumsum_sorted_probs <= 1).to(dtype=torch.float)
            def index_3dtensor_by_2dmask(t, m):
                return torch.gather(t, 2, m.sum(dim=-1, keepdim=True).to(dtype=torch.long))
            t = - (index_3dtensor_by_2dmask(cumsum_sorted_probs, mask) - 1) #/ index_3dtensor_by_2dmask(sorted_probs, mask)
            new_probs = sorted_probs.scatter_add(2, mask.sum(dim=-1, keepdim=True).to(dtype=torch.long), t)
            #t = 1 / index_3dtensor_by_2dmask(cumsum_sorted_probs, mask)
            #new_probs = sorted_probs / index_3dtensor_by_2dmask(cumsum_sorted_probs, mask)
            new_mask = mask.scatter(2, mask.sum(dim=-1).to(dtype=torch.long).unsqueeze(-1),
                                    torch.ones(mask.shape[0], mask.shape[1], 1, device=points.device))
            probs_with_dist = new_mask * new_probs * dist_sorted.unsqueeze(0)
            obj = probs_with_dist.sum(dim=(1, 2))
            """
            obj.mean().backward()
            if opt_idx % 10 == 0 and verbose:
                print(
                    f"idx:{opt_idx} estimated {obj.min():.4f}, {obj.mean():.4f}, best {best_obj:.4f} found at {best_found_at_idx}"
                )
            if sample_num2 is not None and noise2 is not None:
                top_k_indices, probs = gumbel_sinkhorn_topk(
                    gumbel_weights_float,
                    num_clusters,
                    max_iter=100,
                    tau=0.05,
                    sample_num=sample_num2,
                    noise_fact=noise2,
                    return_prob=True,
                )
            cluster_centers = torch.gather(
                torch.repeat_interleave(points.unsqueeze(0), top_k_indices.shape[0], 0),
                1,
                torch.repeat_interleave(
                    top_k_indices.unsqueeze(-1), points.shape[-1], -1
                ),
            )
            obj = compute_objective(
                points.unsqueeze(0), cluster_centers, distance_metric
            )
            best_idx = torch.argmin(obj)
            min_obj, top_k_indices = obj[best_idx], top_k_indices[best_idx]
            # choice_cluster, cluster_centers, selected_indices = discrete_kmeans(
            #    points, num_clusters, init_x=cluster_centers[best_idx], distance=distance_metric, device=points.device)
            # min_obj = compute_objective(points, cluster_centers, distance_metric).item()
            if min_obj < best_obj:
                best_obj = min_obj
                best_top_k_indices = top_k_indices
                best_found_at_idx = opt_idx
            if opt_idx % 10 == 0 and verbose:
                print(
                    f"idx:{opt_idx} real {obj.min():.4f}, {obj.mean():.4f}, best {best_obj:.4f} found at {best_found_at_idx}, now time:{time.time()-prev_time:.2f}"
                )
            optimizer.step()
            optimizer.zero_grad()
            # lr_scheduler.step()
        opt_iter_offset += opt_iters
    cluster_centers = torch.stack(
        [
            torch.gather(points[:, _], 0, best_top_k_indices)
            for _ in range(points.shape[1])
        ],
        dim=-1,
    )
    objective = compute_objective(points, cluster_centers, distance_metric).item()
    # print(f'{index} gumbel objective={objective:.4f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - prev_time}')
    # print(cluster_centers)
    # plt.plot(cluster_centers[:, 0].cpu(), cluster_centers[:, 1].cpu(), 'r+', label='gumbel')

    choice_cluster, cluster_centers, selected_indices = discrete_kmeans(
        points,
        num_clusters,
        init_x=cluster_centers,
        distance=distance_metric,
        device=points.device,
    )
    objective = compute_objective(points, cluster_centers, distance_metric).item()
    # plt.plot(cluster_centers[:, 0].cpu(), cluster_centers[:, 1].cpu(), 'kx', label='gumbel kmeans')
    return objective, selected_indices, time.time() - prev_time


#################################################
#            Traditional FLP Methods            #
#################################################


def initialize(X: Tensor, num_clusters: int, method: str = "plus") -> np.array:
    r"""
    Initialize cluster centers.

    :param X: matrix
    :param num_clusters: number of clusters
    :param method: denotes different initialization strategies: ``'plus'`` (default) or ``'random'``
    :return: initial state

    .. note::
        We support two initialization strategies: random initialization by setting ``method='random'``, or `kmeans++
        <https://en.wikipedia.org/wiki/K-means%2B%2B>`_ by setting ``method='plus'``.
    """
    if method == "plus":
        init_func = _initialize_plus
    elif method == "random":
        init_func = _initialize_random
    else:
        raise NotImplementedError
    return init_func(X, num_clusters)


def _initialize_random(X, num_clusters):
    """
    Initialize cluster centers randomly. See :func:`src.spectral_clustering.initialize` for details.
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def _initialize_plus(X, num_clusters):
    """
    Initialize cluster centers by k-means++. See :func:`src.spectral_clustering.initialize` for details.
    """
    num_samples = len(X)
    centroid_index = np.zeros(num_clusters)
    for i in range(num_clusters):
        if i == 0:
            choice_prob = np.full(num_samples, 1 / num_samples)
        else:
            centroid_X = X[centroid_index[:i]]
            dis = _pairwise_euclidean(X, centroid_X)
            dis_to_nearest_centroid = torch.min(dis, dim=1).values
            choice_prob = dis_to_nearest_centroid / torch.sum(dis_to_nearest_centroid)
            choice_prob = choice_prob.detach().cpu().numpy()

        centroid_index[i] = np.random.choice(
            num_samples, 1, p=choice_prob, replace=False
        )

    initial_state = X[centroid_index]
    return initial_state


def discrete_kmeans(
    X: Tensor,
    num_clusters: int,
    init_x: Union[Tensor, str] = "plus",
    distance: str = "euclidean",
    tol: float = 1e-4,
    device=torch.device("cpu"),
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Perform discrete kmeans on given data matrix :math:`\mathbf X`.
    Here "discrete" means the selected cluster centers must be a subset of the input data :math:`\mathbf X`.

    :param X: :math:`(n\times d)` input data matrix. :math:`n`: number of samples. :math:`d`: feature dimension
    :param num_clusters: (int) number of clusters
    :param init_x: how to initiate x (provide a initial state of x or define a init method) [default: 'plus']
    :param distance: distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: convergence threshold [default: 0.0001]
    :param device: computing device [default: cpu]
    :return: cluster ids, cluster centers, selected indices
    """
    pairwise_distance_function = _get_distance_func(distance)

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if init_x == "rand":
        initial_state = X[torch.randperm(X.shape[0])[:num_clusters], :]
    elif type(init_x) is str:
        initial_state = initialize(X, num_clusters, method=init_x)
    else:
        initial_state = init_x

    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state, device)
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()
        selected_indices = torch.zeros(num_clusters, device=device, dtype=torch.long)
        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index, as_tuple=False).squeeze(
                -1
            )
            if selected.shape[0] == 0:
                continue
            selected_X = torch.index_select(X, 0, selected)
            intra_selected_dist = pairwise_distance_function(
                selected_X, selected_X, device
            )
            index_for_selected = torch.argmin(intra_selected_dist.sum(dim=1))
            initial_state[index] = selected_X[index_for_selected]
            selected_indices[index] = selected[index_for_selected]

        center_shift = torch.sum(
            torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1))
        )

        # increment iteration
        iteration = iteration + 1
        if center_shift**2 < tol:
            break
        if torch.isnan(initial_state).any():
            print("NAN encountered in clustering. Retrying...")
            initial_state = initialize(X, num_clusters)

    return choice_cluster, initial_state, selected_indices


def kmeans(
    X: Tensor,
    num_clusters: int,
    weight: Tensor = None,
    init_x: Union[Tensor, str] = "plus",
    distance: str = "euclidean",
    tol: float = 1e-4,
    device=torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    r"""
    Perform kmeans on given data matrix :math:`\mathbf X`.

    :param X: :math:`(n\times d)` input data matrix. :math:`n`: number of samples. :math:`d`: feature dimension
    :param num_clusters: (int) number of clusters
    :param init_x: how to initiate x (provide a initial state of x or define a init method) [default: 'plus']
    :param distance: distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: convergence threshold [default: 0.0001]
    :param device: computing device [default: cpu]
    :return: cluster ids, cluster centers
    """
    pairwise_distance_function = _get_distance_func(distance)

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if type(init_x) is str:
        initial_state = initialize(X, num_clusters, method=init_x)
    else:
        initial_state = init_x

    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state, device)
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()
        for index in range(num_clusters):
            selected = (
                torch.nonzero(choice_cluster == index, as_tuple=False)
                .squeeze()
                .to(device)
            )
            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)
        center_shift = torch.sum(
            torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1))
        )

        # increase iteration
        iteration = iteration + 1
        if center_shift**2 < tol:
            break
        if torch.isnan(initial_state).any():
            print("NAN encountered in clustering. Retrying...")
            initial_state = initialize(X, num_clusters)

    return choice_cluster, initial_state


def kmeans_predict(
    X: Tensor,
    cluster_centers: Tensor,
    weight: Tensor = None,
    distance: str = "euclidean",
    device=torch.device("cpu"),
) -> Tensor:
    r"""
    Kmeans prediction using existing cluster centers.

    :param X: matrix
    :param cluster_centers: cluster centers
    :param distance: distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: computing device [default: 'cpu']
    :return: cluster ids
    """
    pairwise_distance_function = _get_distance_func(distance)

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers, device)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def _get_distance_func(distance):
    if distance == "euclidean":
        return _pairwise_euclidean
    elif distance == "cosine":
        return _pairwise_cosine
    elif distance == "manhattan":
        return _pairwise_manhattan
    else:
        raise NotImplementedError


def _pairwise_euclidean(data1, data2, device=torch.device("cpu")):
    """Compute pairwise Euclidean distance"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=-2)

    # 1*N*M
    B = data2.unsqueeze(dim=-3)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1)
    return dis


def _pairwise_manhattan(data1, data2, device=torch.device("cpu")):
    """Compute pairwise Manhattan distance"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=-2)

    # 1*N*M
    B = data2.unsqueeze(dim=-3)

    dis = torch.abs(A - B)
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1)
    return dis


def _pairwise_cosine(data1, data2, device=torch.device("cpu")):
    """Compute pairwise cosine distance"""
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=-2)

    # 1*N*M
    B = data2.unsqueeze(dim=-3)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze(-1)
    return cosine_dis


def spectral_clustering(
    sim_matrix: Tensor,
    cluster_num: int,
    init: Tensor = None,
    return_state: bool = False,
    normalized: bool = False,
):
    r"""
    Perform spectral clustering based on given similarity matrix.

    This function firstly computes the leading eigenvectors of the given similarity matrix, and then utilizes the
    eigenvectors as features and performs k-means clustering based on these features.

    :param sim_matrix: :math:`(n\times n)` input similarity matrix. :math:`n`: number of instances
    :param cluster_num: number of clusters
    :param init: the initialization technique or initial features for k-means
    :param return_state: whether return state features (can be further used for prediction)
    :param normalized: whether to normalize the similarity matrix by its degree
    :return: the belonging of each instance to clusters, state features (if ``return_state==True``)
    """
    degree = torch.diagflat(torch.sum(sim_matrix, dim=-1))
    if normalized:
        aff_matrix = (degree - sim_matrix) / torch.diag(degree).unsqueeze(1)
    else:
        aff_matrix = degree - sim_matrix
    e, v = torch.symeig(aff_matrix, eigenvectors=True)
    topargs = torch.argsort(torch.abs(e), descending=False)[1:cluster_num]
    v = v[:, topargs]

    if cluster_num == 2:
        choice_cluster = (v > 0).to(torch.int).squeeze(1)
    else:
        choice_cluster, initial_state = kmeans(
            v,
            cluster_num,
            init_x=init if init is not None else "plus",
            distance="euclidean",
            tol=1e-6,
        )

    choice_cluster = choice_cluster.to(sim_matrix.device)

    if return_state:
        return choice_cluster, initial_state
    else:
        return choice_cluster


def greedy_facility_location(
    X: Tensor,
    num_clusters: int,
    weight: Tensor = None,
    distance: str = "euclidean",
    device=torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    r"""
    Greedy algorithm for facility location problem.
    This is function also solves the discrete clustering problem given data matrix :math:`\mathbf X`.
    Here "discrete" means the selected cluster centers must be a subset of the input data :math:`\mathbf X`.

    :param X: :math:`(n\times d)` input data matrix. :math:`n`: number of samples. :math:`d`: feature dimension
    :param num_clusters: (int) number of clusters
    :param distance: distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: computing device [default: cpu]
    :return: cluster centers, selected indices
    """
    pairwise_distance_function = _get_distance_func(distance)

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    selected_indices = []
    unselected_indices = list(range(X.shape[0]))

    for cluster_center_idx in range(num_clusters):
        best_dis = float("inf")
        best_idx = -1

        dis_list = torch.zeros(X.shape[0]).to(device)

        for unselected_idx in unselected_indices:
            selected = torch.tensor(selected_indices + [unselected_idx], device=device)
            selected_X = torch.index_select(X, 0, selected)
            dis = pairwise_distance_function(X, selected_X, device)
            nearest_dis = dis.min(dim=1).values.sum()
            dis_list[unselected_idx] = nearest_dis
            if nearest_dis < best_dis:
                best_dis = nearest_dis
                best_idx = unselected_idx

        unselected_indices.remove(best_idx)
        selected_indices.append(best_idx)

    selected_indices = torch.tensor(selected_indices, device=device)
    cluster_centers = torch.index_select(X, 0, selected_indices)
    return cluster_centers, selected_indices


def ortools_facility_location(
    X: Tensor,
    num_clusters: int,
    distance: str = "euclidean",
    solver_name=None,
    linear_relaxation=True,
    timeout_sec=60,
):
    # define solver instance
    if solver_name is None:
        if linear_relaxation:
            solver = pywraplp.Solver(
                "facility_location", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING
            )
        else:
            solver = pywraplp.Solver(
                "facility_location", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
            )
    else:
        solver = pywraplp.Solver.CreateSolver(solver_name)

    X = X.cpu()

    # Initialize variables
    VarX = {}
    VarY = {}
    ConstY1 = {}
    ConstY2 = {}
    for selected_id in range(X.shape[0]):
        if linear_relaxation:
            VarX[selected_id] = solver.NumVar(0.0, 1.0, f"x_{selected_id}")
        else:
            VarX[selected_id] = solver.BoolVar(f"x_{selected_id}")

        VarY[selected_id] = {}
        for all_point_id in range(X.shape[0]):
            if linear_relaxation:
                VarY[selected_id][all_point_id] = solver.NumVar(
                    0.0, 1.0, f"y_{selected_id}_{all_point_id}"
                )
            else:
                VarY[selected_id][all_point_id] = solver.BoolVar(
                    f"y_{selected_id}_{all_point_id}"
                )

    # Constraints
    X_constraint = 0
    for selected_id in range(X.shape[0]):
        X_constraint += VarX[selected_id]
    solver.Add(X_constraint <= num_clusters)

    for selected_id in range(X.shape[0]):
        ConstY1[selected_id] = 0
        for all_point_id in range(X.shape[0]):
            ConstY1[selected_id] += VarY[selected_id][all_point_id]
        solver.Add(ConstY1[selected_id] <= VarX[selected_id] * X.shape[0])

    for all_point_id in range(X.shape[0]):
        ConstY2[all_point_id] = 0
        for selected_id in range(X.shape[0]):
            ConstY2[all_point_id] += VarY[selected_id][all_point_id]
        solver.Add(ConstY2[all_point_id] == 1)

    # The distance
    pairwise_distance_function = _get_distance_func(distance)
    distance_matrix = pairwise_distance_function(X, X)

    # the objective
    distance = 0
    for selected_id in range(X.shape[0]):
        for all_point_id in range(X.shape[0]):
            distance += (
                distance_matrix[selected_id][all_point_id].item()
                * VarY[selected_id][all_point_id]
            )

    solver.Minimize(distance)

    if timeout_sec > 0:
        solver.set_time_limit(int(timeout_sec * 1000))
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        return solver.Objective().Value(), [
            VarX[_].solution_value() for _ in range(X.shape[0])
        ]
    else:
        print(
            "The problem does not have an optimal solution. status={}.".format(status)
        )
        return solver.Objective().Value(), [
            VarX[_].solution_value() for _ in range(X.shape[0])
        ]


def gurobi_facility_location(
    X: Tensor,
    num_clusters: int,
    distance: str = "euclidean",
    linear_relaxation=True,
    timeout_sec=60,
    start=None,
    verbose=True,
    seed=42,
):
    import gurobipy as grb

    try:
        model = grb.Model("facility location")        
        if verbose:
            model.setParam("LogToConsole", 1)
        else:
            model.setParam("LogToConsole", 0)
        # model.setParam('MIPFocus', 1)
        if timeout_sec > 0:
            model.setParam("TimeLimit", timeout_sec)
        model.setParam("Seed", seed)

        X = X.cpu()

        # Initialize variables
        VarX = {}
        VarY = {}
        ConstY1 = {}
        ConstY2 = {}
        for selected_id in range(X.shape[0]):
            if linear_relaxation:
                VarX[selected_id] = model.addVar(
                    0.0, 1.0, vtype=grb.GRB.CONTINUOUS, name=f"x_{selected_id}"
                )
            else:
                VarX[selected_id] = model.addVar(
                    vtype=grb.GRB.BINARY, name=f"x_{selected_id}"
                )
            if start is not None:
                VarX[selected_id].start = start[selected_id]
            VarY[selected_id] = {}
            for all_point_id in range(X.shape[0]):
                if linear_relaxation:
                    VarY[selected_id][all_point_id] = model.addVar(
                        0.0,
                        1.0,
                        vtype=grb.GRB.CONTINUOUS,
                        name=f"y_{selected_id}_{all_point_id}",
                    )
                else:
                    VarY[selected_id][all_point_id] = model.addVar(
                        vtype=grb.GRB.BINARY, name=f"y_{selected_id}_{all_point_id}"
                    )

        # Constraints
        X_constraint = 0
        for selected_id in range(X.shape[0]):
            X_constraint += VarX[selected_id]
        model.addConstr(X_constraint <= num_clusters)
        for selected_id in range(X.shape[0]):
            ConstY1[selected_id] = 0
            for all_point_id in range(X.shape[0]):
                ConstY1[selected_id] += VarY[selected_id][all_point_id]
            model.addConstr(ConstY1[selected_id] <= VarX[selected_id] * X.shape[0])
        for all_point_id in range(X.shape[0]):
            ConstY2[all_point_id] = 0
            for selected_id in range(X.shape[0]):
                ConstY2[all_point_id] += VarY[selected_id][all_point_id]
            model.addConstr(ConstY2[all_point_id] == 1)

        # The distance
        pairwise_distance_function = _get_distance_func(distance)
        distance_matrix = pairwise_distance_function(X, X)

        # the objective
        distance = 0
        for selected_id in range(X.shape[0]):
            for all_point_id in range(X.shape[0]):
                distance += (
                    distance_matrix[selected_id][all_point_id].item()
                    * VarY[selected_id][all_point_id]
                )
        model.setObjective(distance, grb.GRB.MINIMIZE)

        model.optimize()

        res = [model.getVarByName(f"x_{set_id}").X for set_id in range(X.shape[0])]
        if linear_relaxation:
            res = np.array(res, dtype=np.float)
        else:
            res = np.array(res, dtype=np.int)
        return model.getObjective().getValue(), torch.from_numpy(res).to(X.device)

    except grb.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))

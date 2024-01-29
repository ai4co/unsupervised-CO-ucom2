from itertools import product
from src.max_covering_methods import *
import time
import xlwt
from datetime import datetime
import os
from src.config import load_config, load_config_egnpb_max_covering
from src.max_covering_data import *
from src.poi_bin import *
from tqdm import trange, tqdm

####################################
#             config               #
####################################

# cfg = load_config()
args, cfg = load_config_egnpb_max_covering()
device = torch.device("cuda:0")

timestamp_ = args.timestamp

def seed_everything(seed: int):
    # https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

####################################
#            training              #
####################################

if cfg.train_data_type == "random":
    train_dataset = get_random_dataset(cfg.num_items, cfg.num_sets, 1)
elif cfg.train_data_type == "twitch":
    train_dataset = get_twitch_dataset()
else:
    raise ValueError(f"Unknown training dataset {cfg.train_data_type}!")

model = GNNModel().to(device)

from time import strftime, localtime


def get_local_time():
    return strftime("%Y%m%d%H%M%S", localtime())

timestamp_true = get_local_time()

lr_ = args.lr
lr_string = str(lr_).replace(".", "")
reg_ = args.reg
reg_string = str(int(reg_))

model_path = f"saved_models/max_covering_pb_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_saved_{timestamp_}.pt"

if not os.path.exists(model_path):
    print(f"Training the model weights")
    train_optimizer = torch.optim.Adam(model.parameters(), lr=lr_)
    # training loop
    print(f"learning rate = {lr_}; reg coef = {reg_}")
    graph_list = []
    bipartite_adj_list = []
    for name, weights, sets in train_dataset:
        graph_list.append(build_graph_from_weights_sets(weights, sets, device))        
        bipartite_adj_list.append(compute_bipartite_adj(weights, sets, device))
        
    for epoch in trange(100):
        obj_sum = 0
        for index, (name, weights, sets) in enumerate(train_dataset):
            bipartite_adj = bipartite_adj_list[index]
            graph = graph_list[index]
            probs = model(graph)
            card_dist = pmf_poibin_vec(probs, device, use_normalization=False)
            k_diff = torch.abs(
                torch.arange(probs.shape[0] + 1, device=device)
                - cfg.train_max_covering_items
            )
            constraint_conflict = (card_dist * k_diff).sum()
            obj, _ = compute_obj_differentiable_fixed(
                weights, sets, probs, bipartite_adj, device=probs.device
            )
            obj = obj - reg_ * constraint_conflict
            (-obj).mean().backward()
            obj_sum += obj.mean()
            train_optimizer.step()
            train_optimizer.zero_grad()
        # print(f'epoch {epoch}, obj={obj_sum / len(train_dataset)}')
    torch.save(model.state_dict(), model_path)
    # print(f'Model saved to {model_path}.')
else:
    # warm up
    name, weights, sets = train_dataset[0]
    graph = build_graph_from_weights_sets(weights, sets, device)    
    probs = model(graph)
    card_dist = pmf_poibin_vec(probs, device, use_normalization=False)
    print("########## Loading existing model ##########")

####################################
#            testing               #
####################################

if cfg.test_data_type == "random":
    dataset = get_random_dataset(cfg.num_items, cfg.num_sets, 0)
elif cfg.test_data_type == "twitch":
    dataset = get_twitch_dataset()
elif cfg.test_data_type == "rail":
    dataset = get_rail_dataset()
else:
    raise ValueError(f"Unknown testing dataset {cfg.test_data_type}!")

wb = xlwt.Workbook()
ws = wb.add_sheet(
    f"max_covering_{cfg.test_max_covering_items}-{cfg.num_sets}-{cfg.num_items}"
)
ws.write(0, 0, "name")
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
timestamp_true = get_local_time()
p_save = Path("res_ours_mc_0118")
p_save.mkdir(exist_ok=True)
res_save_name = p_save / f"max_covering-ucom2-{timestamp_true}-{cfg.test_data_type}_{cfg.test_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_lr{lr_string}_reg{reg_string}_{timestamp_}.xls"

graph_list = []
bipartite_adj_list = []
for index, (name, weights, sets) in enumerate(dataset):
    graph_list.append(build_graph_from_weights_sets(weights, sets, device))
    bipartite_adj_list.append(compute_bipartite_adj(weights, sets, device))


if cfg.test_data_type == "random":
    if cfg.num_items == 1000:
        trail_round_list = [(1, 1), (2, 1), (5, 1), (10, 1), (20, 1), (50, 1), (100, 1), (200, 1), (200, 2)]
    elif cfg.num_items == 2000:
        trail_round_list = [(1, 1), (2, 1), (5, 1), (10, 1), (20, 1), (50, 1), (100, 1), (100, 2)]
elif cfg.test_data_type == "twitch":
    trail_round_list = [(1, 1), (1, 2), (1, 5), (1, 10)]
elif cfg.test_data_type == "rail":
    trail_round_list = [(1, 1), (2, 1), (5, 1), (10, 1), (10, 2)]


for index, (name, weights, sets) in enumerate(dataset):    
    method_idx = 0
    print("-" * 20)
    print(f"{name} items={len(weights)} sets={len(sets)}")
    ws.write(index + 1, 0, name)

    graph = graph_list[index]
    bipartite_adj = bipartite_adj_list[index]   
    
    for trials_, num_rounds in trail_round_list:                     
        model.load_state_dict(torch.load(model_path))         
        res_list = egn_pb_greedy_faster_max_covering_full_truncated(        
            weights,
            sets,
            graph,
            bipartite_adj,
            max_covering_items=cfg.test_max_covering_items,
            model=model,
            egn_beta=reg_,
            random_trials=trials_,
            num_rounds=num_rounds,
            time_limit=-1,
            noise_scale=0.2,
            stru_noise=0.2,
            s_max=100,
            zero_start=cfg.test_data_type == "twitch",
        )
        
        for i_round in range(num_rounds):
            objective, best_top_k_indices, finish_time = res_list[i_round]
            method_idx += 1
            if index == 0:            
                ws.write(0, method_idx * 2 - 1, f"{i_round + 1}*{trials_}-objective")
                ws.write(0, method_idx * 2, f"{i_round + 1}*{trials_}-time")
            ws.write(index + 1, method_idx * 2 - 1, objective)
            ws.write(index + 1, method_idx * 2, finish_time)
        wb.save(res_save_name)

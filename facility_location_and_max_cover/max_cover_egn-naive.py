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

lr_ = args.lr
lr_string = str(lr_).replace(".", "")
reg_ = args.reg
reg_string = str(int(reg_))

timestamp_ = args.timestamp
model_path = f"saved_models/max_covering_egn-naive_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_saved_lr{lr_string}_reg{reg_string}_{timestamp_}.pt"

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
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}.')
else:    
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

p_save = Path("res_egn_naive_mc_0118")
p_save.mkdir(parents=True, exist_ok=True)
output_file_name = p_save / f"max_covering_{cfg.test_max_covering_items}-{cfg.num_sets}-{cfg.num_items}-{cfg.test_data_type}_{timestamp_}_{timestamp_true}.txt"

for index, (name, weights, sets) in enumerate(dataset):
    model.load_state_dict(torch.load(model_path))
    obj_list, time_list = egn_max_covering(        
        weights,
        sets,            
        max_covering_items=cfg.test_max_covering_items,
        model=model,
        egn_beta=reg_,
        random_trials=1000,
        time_limit=120,
    )
    print(name, obj_list[-1], time_list[-1])
    # print into the output file too
    with open(output_file_name, "a") as f:
        f.write(f"{name} {obj_list[-1]} {time_list[-1]}\n")


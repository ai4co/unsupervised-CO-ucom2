from pathlib import Path
from src.facility_location_methods import *
import time
from time import strftime, localtime
import xlwt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import os
from src.facility_location_data import *
from src.config import load_config_egnpb_facility_location

####################################
#             config               #
####################################

args, cfg = load_config_egnpb_facility_location()
device = torch.device("cuda:0")


def get_local_time():
    return strftime("%Y%m%d%H%M%S", localtime())

if not args.timestamp:
    timestamp_ = get_local_time()
else:
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

try:
    i_seed = int(args.timestamp)
except:
    i_seed = hash(args.timestamp) % (2 ** 32)    
seed_everything(i_seed)
    

wb = xlwt.Workbook()
ws = wb.add_sheet(
    f"clustering_{cfg.test_data_type}_{cfg.test_num_centers}-{cfg.num_data}"
)
ws.write(0, 0, "name")
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


####################################
#            training              #
####################################

if cfg.train_data_type == "random":
    train_dataset = get_random_data(cfg.num_data, cfg.dim, 0, device)
elif cfg.train_data_type == "starbucks":
    train_dataset = get_starbucks_data(device)
else:
    raise ValueError(f"Unknown dataset name {cfg.train_dataset_type}!")

model = GNNModel().to(device)

from time import strftime, localtime


def get_local_time():
    return strftime("%Y%m%d%H%M%S", localtime())


def param2string(param_):
    return str(param_).replace(".", "p")


lr_ = args.lr
lr_string = param2string(lr_)
reg_ = args.reg
reg_string = param2string(reg_)

if args.timestamp:
    timestamp_ = args.timestamp
else:
    timestamp_ = get_local_time()

model_path = f"saved_models/facility_location_egn-naive_{cfg.train_data_type}_{cfg.train_num_centers}-{cfg.num_data}_saved_{timestamp_}.pt"

model = GNNModel().to(device)
train_outer_optimizer = torch.optim.Adam(model.parameters(), lr=lr_)
# training loop
if not os.path.exists(model_path):
    print("########## training ##########")
    graph_train_list = []
    dist_train_list = []
    for index, (_, points) in enumerate(train_dataset):
        graph, dist = build_graph_from_points(points, None, True, cfg.distance_metric)
        graph_train_list.append(graph)
        dist_train_list.append(dist)
    for epoch in trange(100):
        obj_sum = 0
        for index, (_, points) in enumerate(train_dataset):
            graph = graph_train_list[index]
            dist = dist_train_list[index]
            probs = model(graph)
            constraint_conflict = torch.relu(probs.sum() - cfg.train_num_centers)
            obj = (
                compute_objective_differentiable_exact(dist, probs)
                + reg_ * constraint_conflict
            )
            obj.mean().backward()
            obj_sum += obj.mean()
            train_outer_optimizer.step()
            train_outer_optimizer.zero_grad()
    torch.save(model.state_dict(), model_path)

####################################
#            testing               #
####################################

if cfg.test_data_type == "random":
    dataset = get_random_data(cfg.num_data, cfg.dim, 1, device)
elif cfg.test_data_type == "starbucks":
    dataset = get_starbucks_data(device)
elif cfg.test_data_type == "mcd":
    dataset = get_mcd_data(device)
elif cfg.test_data_type == "subway":
    dataset = get_subway_data(device)
else:
    raise ValueError(f"Unknown dataset name {cfg.test_data_type}!")

# breakpoint()

graph_test_list = []
dist_test_list = []
for index, (prob_name, points) in enumerate(dataset):
    graph, dist = build_graph_from_points(points, None, True, cfg.distance_metric)
    graph_test_list.append(graph)
    dist_test_list.append(dist)

timestamp_now = get_local_time()

for index, (prob_name, points) in enumerate(dataset):
    method_idx = 0
    graph = graph_test_list[index]
    dist = dist_test_list[index]
    print("-" * 20)
    print(f"{prob_name} points={len(points)} select={cfg.test_num_centers}")
    ws.write(index + 1, 0, prob_name)
   
       
    method_idx += 1
    model.load_state_dict(torch.load(model_path))
    objective, selected_indices, finish_time = egn_facility_location_exact(
        points,
        cfg.test_num_centers,
        model,
        egn_beta=reg_,
        random_trials=1000,
        time_limit=120,
        distance_metric="euclidean",
    )
    ws.write(index + 1, method_idx * 2 - 1, objective)
    ws.write(index + 1, method_idx * 2, finish_time)
p_save = Path("res_egn_naive_0118")
p_save.mkdir(exist_ok=True)
wb.save(
    p_save / f"facility_location-egn-naive-{cfg.test_data_type}-{cfg.test_num_centers}-{cfg.num_data}-{timestamp_}-lr{lr_string}-reg{reg_string}-{timestamp_}-{timestamp_now}.xls")

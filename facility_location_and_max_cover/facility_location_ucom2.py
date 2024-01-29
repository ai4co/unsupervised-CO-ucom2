from itertools import product
from pathlib import Path
from src.facility_location_methods import *
import time
import xlwt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import os
from src.facility_location_data import *
from src.config import load_config, load_config_egnpb_facility_location
from src.poi_bin import pmf_poibin, pmf_poibin_vec

####################################
#             config               #
####################################

args, cfg = load_config_egnpb_facility_location()
device = torch.device("cuda:0")

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

model_path = f"saved_models/facility_location_ucom2_{cfg.train_data_type}_{cfg.train_num_centers}-{cfg.num_data}_saved_{timestamp_}.pt"

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
            # graph, dist = build_graph_from_points(points, None, True, cfg.distance_metric)
            graph = graph_train_list[index]
            dist = dist_train_list[index]
            probs = model(graph)
            card_dist = pmf_poibin_vec(probs, device, use_normalization=False)
            k_diff = torch.abs(
                torch.arange(probs.shape[0] + 1, device=device) - cfg.train_num_centers
            )
            # avoid empty output
            k_diff[:2] *= 10
            constraint_conflict = (card_dist * k_diff).sum()
            obj = (
                compute_objective_differentiable_exact(dist, probs)
                + reg_ * constraint_conflict
            )
            obj.mean().backward()
            obj_sum += obj.mean()
            train_outer_optimizer.step()
            train_outer_optimizer.zero_grad()
        # print(f'epoch {epoch}, obj={obj_sum / len(train_dataset)}')
    torch.save(model.state_dict(), model_path)
    # print(f'Model saved to {model_path}.')
else:
    # warm up
    _, points = train_dataset[0]
    graph, dist = build_graph_from_points(points, None, True, cfg.distance_metric)
    probs = model(graph)
    card_dist = pmf_poibin_vec(probs, device, use_normalization=False)
    # print(f'epoch {epoch}, obj={obj_sum / len(train_dataset)}')    
    print("########## Loading existing model ##########")


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


graph_test_list = []
dist_test_list = []
for index, (prob_name, points) in enumerate(dataset):
    graph, dist = build_graph_from_points(points, None, True, cfg.distance_metric)
    graph_test_list.append(graph)
    dist_test_list.append(dist)

timestamp_now = get_local_time()
best_objective_list_list = []


if cfg.test_data_type == "random" and cfg.num_data == 500:
    trial_round_list = [(1, 1), (5, 1), (10, 1), (20, 1), (50, 1), (100, 1), (200, 1), (500, 1), (500, 2)]
elif cfg.test_data_type == "random" and cfg.num_data == 800:
    trial_round_list = [(1, 1), (5, 1), (10, 1), (20, 1), (50, 1), (100, 1), (200, 1), (200, 2)]
elif cfg.test_data_type == "starbucks":
    trial_round_list = [(1, 1), (5, 1), (10, 1), (20, 1), (50, 1), (50, 2)]
elif cfg.test_data_type == "mcd":
    trial_round_list = [(1, 1), (5, 1), (10, 1), (20, 1), (50, 1), (50, 2)]
elif cfg.test_data_type == "subway":
    trial_round_list = [(1, 1), (5, 1), (10, 1), (10, 2)]
       

for index, (prob_name, points) in enumerate(dataset):
    method_idx = 0
    graph = graph_test_list[index]
    dist = dist_test_list[index]
    # print("-" * 20)
    # print(f"{prob_name} points={len(points)} select={cfg.test_num_centers}")
    ws.write(index + 1, 0, prob_name)    
    noise_ = 0.2
    stru_ = 0.2

    for i_tr, (trials_, rounds_) in enumerate(trial_round_list):
        method_idx += 1
        model.load_state_dict(torch.load(model_path))
        print("model loaded")
        (
            objective,
            selected_indices,
            finish_time,
            best_objective_list,
        ) = egn_pb_greedy_facility_location_exact_fast_test_truncated_soft_then_hard(
            points,
            graph,
            dist,
            cfg.test_num_centers,
            model,
            egn_beta=reg_,
            random_trials=trials_,
            num_rounds=rounds_,
            time_limit=-1,
            noise_scale=noise_,
            stru_noise=stru_,
            distance_metric="euclidean",
            s_max=100,
            hard_after=True,
        )
        if index == 0:
            ws.write(0, method_idx * 2 - 1, f"{rounds_}*{trials_}-{noise_}-{stru_}-objective")
            ws.write(0, method_idx * 2, f"{rounds_}*{trials_}-{noise_}-{stru_}-time")
        ws.write(index + 1, method_idx * 2 - 1, objective)
        ws.write(index + 1, method_idx * 2, finish_time)            
        print(datetime.now(), objective, finish_time)
p_save = Path("res_ours_0118")
p_save.mkdir(exist_ok=True)
wb.save(
    p_save / f"facility_location-ucom2-{cfg.test_data_type}-{cfg.test_num_centers}-{cfg.num_data}-{timestamp_}-lr{lr_string}-reg{reg_string}-{timestamp_now}.xls"
)


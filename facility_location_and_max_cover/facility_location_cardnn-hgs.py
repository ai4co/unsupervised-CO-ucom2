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

test_timestamp = None

lr_ = args.lr
lr_string = str(lr_).replace(".", "")
reg_ = args.reg
reg_string = str(reg_).replace(".", "")

cfg.gumbel_sigma = reg_
for i in range(len(cfg.homotophy_sigma)):
    cfg.homotophy_sigma[i] = reg_
    
model_path = f"saved_models/facility_location_cardnn-hgs_{cfg.train_data_type}_{cfg.train_num_centers}-{cfg.num_data}_saved_{timestamp_}.pt"

if not os.path.exists(model_path):
    print('Training the model weights')
    model = GNNModel().to(device)            
    train_outer_optimizer = torch.optim.Adam(model.parameters(), lr=lr_)
    for epoch in range(cfg.train_iter):    
        # training loop
        obj_sum = 0
        for index, (_, points) in enumerate(train_dataset):
            graph, dist = build_graph_from_points(points, None, True, cfg.distance_metric)
            latent_vars = model(graph)
            
            sample_num = cfg.gumbel_sample_num
            noise_fact = cfg.gumbel_sigma
            
            top_k_indices, probs = gumbel_sinkhorn_topk(
                latent_vars, cfg.train_num_centers, max_iter=100, tau=.05,
                sample_num=sample_num, noise_fact=reg_, return_prob=True
            )
            # compute objective by softmax
            obj = compute_objective_differentiable(dist, probs, temp=50) # set smaller temp during training
            obj.mean().backward()
            obj_sum += obj.mean()
            train_outer_optimizer.step()
            train_outer_optimizer.zero_grad()
        print(f'epoch {epoch}, obj={obj_sum / len(train_dataset)}')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}.')

####################################
#            testing               #
####################################

if cfg.test_data_type == 'random':
    dataset = get_random_data(cfg.num_data, cfg.dim, 1, device)
elif cfg.test_data_type == 'starbucks':
    dataset = get_starbucks_data(device)
elif cfg.test_data_type == 'mcd':
    dataset = get_mcd_data(device)
elif cfg.test_data_type == 'subway':
    dataset = get_subway_data(device)
else:
    raise ValueError(f'Unknown dataset name {cfg.train_dataset_type}!')

graph_test_list = []
dist_test_list = []

for index, (prob_name, points) in enumerate(dataset):
    graph, dist = build_graph_from_points(points, None, True, cfg.distance_metric)
    graph_test_list.append(graph)
    dist_test_list.append(dist)

timestamp_true = get_local_time()
for index, (prob_name, points) in enumerate(dataset):
    method_idx = 0
    graph = graph_test_list[index]
    dist = dist_test_list[index]
    # print('-' * 20)
    # print(f'{prob_name} points={len(points)} select={cfg.test_num_centers}')
    ws.write(index+1, 0, prob_name)

    # original CardNN-HGS   
    method_idx += 1
    model.load_state_dict(torch.load(model_path))
    objective, selected_indices, finish_time = sinkhorn_facility_location(
        points, cfg.test_num_centers, model,
        cfg.softmax_temp, cfg.gumbel_sample_num, cfg.homotophy_sigma, cfg.homotophy_tau, cfg.homotophy_sk_iter, cfg.homotophy_opt_iter,
        time_limit=-1, 
        # verbose=cfg.verbose,
        verbose=True, 
        distance_metric=cfg.distance_metric,
        )
    print(f'{prob_name} CardNN-HGS objective={objective:.4f} selected={sorted(selected_indices.cpu().numpy().tolist())} time={finish_time}')
    if index == 0:
        ws.write(0, method_idx * 2 - 1, 'CardNN-HGS-objective')
        ws.write(0, method_idx * 2, 'CardNN-HGS-time')
    ws.write(index + 1, method_idx * 2 - 1, objective)
    ws.write(index + 1, method_idx * 2, finish_time)

wb.save(f'res/facility_location-cardnn-hgs-{cfg.test_data_type}-{cfg.test_num_centers}-{timestamp_true}.xls')

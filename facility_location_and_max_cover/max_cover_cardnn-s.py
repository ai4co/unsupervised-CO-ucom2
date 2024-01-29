from src.max_covering_methods import *
import time
import xlwt
from datetime import datetime
import os
from src.config import load_config, load_config_egnpb_max_covering
from src.max_covering_data import *
from tqdm import trange, tqdm

####################################
#             config               #
####################################

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

try:
    i_seed = int(args.timestamp)
except:
    i_seed = hash(args.timestamp) % (2 ** 32)
seed_everything(i_seed)


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
try:
    reg_string = str(int(reg_))
except:
    reg_string = str(float(reg_))

cfg.gumbel_sigma = reg_
for i in range(len(cfg.homotophy_sigma)):
    cfg.homotophy_sigma[i] = reg_
cfg.train_lr = lr_

model_path = f"saved_models/max_covering_cardnn-s_{cfg.train_data_type}_{cfg.train_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_{timestamp_}.pt"

if not os.path.exists(model_path):
    print(f"Training the model weights")
    model = GNNModel().to(device)
    train_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_lr)
    for epoch in trange(cfg.train_iter):
        # training loop
        obj_sum = 0
        for name, weights, sets in train_dataset:
            bipartite_adj = None
            graph = build_graph_from_weights_sets(weights, sets, device)
            latent_vars = model(graph)
            sample_num = cfg.train_gumbel_sample_num
            noise_fact = cfg.gumbel_sigma
            top_k_indices, probs = gumbel_sinkhorn_topk(
                latent_vars,
                cfg.train_max_covering_items,
                max_iter=cfg.sinkhorn_iter,
                tau=cfg.sinkhorn_tau,
                sample_num=sample_num,
                noise_fact=noise_fact,
                return_prob=True,
            )
            # compute objective by softmax
            obj, _ = compute_obj_differentiable(
                weights, sets, probs, bipartite_adj, device=probs.device
            )
            (-obj).mean().backward()
            obj_sum += obj.mean()

            train_optimizer.step()
            train_optimizer.zero_grad()

        print(f"epoch {epoch}, obj={obj_sum / len(train_dataset)}")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}.")


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


for index, (name, weights, sets) in enumerate(dataset):
    method_idx = 0
    print("-" * 20)
    print(f"{name} items={len(weights)} sets={len(sets)}")
    ws.write(index + 1, 0, name)

    # CardNN-S
    method_idx += 1
    prev_time = time.time()
    model.load_state_dict(torch.load(model_path))
    # best_obj, best_top_k_indices = sinkhorn_max_covering(weights, sets, cfg.test_max_covering_items, model, cfg.gumbel_sample_num, cfg.homotophy_sigma, cfg.homotophy_tau, cfg.homotophy_sk_iter, cfg.homotophy_opt_iter, verbose=cfg.verbose)
    best_obj, best_top_k_indices = sinkhorn_max_covering_fast(
        weights,
        sets,
        cfg.test_max_covering_items,
        model,
        1,
        0,
        cfg.sinkhorn_tau,
        cfg.sinkhorn_iter,
        cfg.soft_opt_iter,
        verbose=cfg.verbose,
    )
    # print(f'{name} CardNN-S objective={best_obj:.0f} selected={sorted(best_top_k_indices.cpu().numpy().tolist())} time={time.time() - prev_time}')
    if index == 0:
        ws.write(0, method_idx * 2 - 1, "CardNN-S-objective")
        ws.write(0, method_idx * 2, "CardNN-S-time")
    try:
        ws.write(index + 1, method_idx * 2 - 1, best_obj.item())
    except:
        ws.write(index + 1, method_idx * 2 - 1, float(best_obj))
    ws.write(index + 1, method_idx * 2, time.time() - prev_time)
wb.save(
    f"res/max_covering-cardnn-s-{cfg.test_data_type}_{cfg.test_max_covering_items}-{cfg.num_sets}-{cfg.num_items}_{get_local_time()}.xls"
)

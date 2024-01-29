from src.max_covering_methods import compute_bipartite_adj, compute_objective
import time
from src.config import load_config_egnpb_facility_location
from src.max_covering_data import *
from tqdm import tqdm, trange

args, cfg = load_config_egnpb_facility_location()
device = torch.device("cpu")

if cfg.test_data_type == "random":
    dataset = get_random_dataset(cfg.num_items, cfg.num_sets, 0)
elif cfg.test_data_type == "twitch":
    dataset = get_twitch_dataset()
elif cfg.test_data_type == "rail":
    dataset = get_rail_dataset()
else:
    raise ValueError(f"Unknown testing dataset {cfg.test_data_type}!")


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
    i_seed = time.time_ns()
seed_everything(i_seed)


for index, (name, weights, sets) in tqdm(
    enumerate(dataset), total=len(dataset), position=0
):
    method_idx = 0
    n_random_max = int(1e10)
    bipartite_adj = compute_bipartite_adj(weights, sets, device=device)
    best_obj = 0.0
    prev_time = time.time()
    for i_trial in trange(n_random_max, position=1, leave=False):
        if time.time() - prev_time > 240:
            break
        perm = torch.randperm(len(sets))
        idx = perm[: cfg.test_max_covering_items]
        objective = compute_objective(
            weights, sets, idx, bipartite_adj, device=device
        ).item()
        best_obj = max(best_obj, objective)
    print(name, best_obj, time.time() - prev_time)

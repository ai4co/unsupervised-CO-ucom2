from src.max_covering_methods import ortools_max_covering
import time
from src.config import load_config_egnpb_facility_location
from src.max_covering_data import *

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


for index, (name, weights, sets) in enumerate(dataset):
    time_budget = 120
    prev_time = time.time()
    ip_obj, ip_scores = ortools_max_covering(
        weights,
        sets,
        cfg.test_max_covering_items,
        solver_name="SCIP",
        linear_relaxation=False,
        timeout_sec=time_budget,
    )
    print(name, ip_obj, time.time() - prev_time)

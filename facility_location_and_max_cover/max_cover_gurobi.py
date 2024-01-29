from src.max_covering_methods import gurobi_max_covering
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

try:
    i_seed = int(args.timestamp)
except:
    i_seed = time.time_ns()

for index, (name, weights, sets) in enumerate(dataset):
    time_budget = 120
    prev_time = time.time()
    ip_obj, ip_scores = gurobi_max_covering(
        weights,
        sets,
        cfg.test_max_covering_items,
        linear_relaxation=False,
        timeout_sec=time_budget,
        verbose=cfg.verbose,
        seed=i_seed,
    )
    print(name, ip_obj, time.time() - prev_time)

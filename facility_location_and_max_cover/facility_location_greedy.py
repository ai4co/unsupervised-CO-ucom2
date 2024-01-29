from tqdm import tqdm, trange
from src.facility_location_methods import compute_objective, greedy_facility_location
import time
from time import strftime, localtime
from src.facility_location_data import *
from src.config import load_config_egnpb_facility_location

####################################
#             config               #
####################################

args, cfg = load_config_egnpb_facility_location()
device = torch.device("cpu")

if cfg.test_data_type == "random":
    dataset = get_random_data(cfg.num_data, cfg.dim, 1, device)
elif cfg.test_data_type == "starbucks":
    dataset = get_starbucks_data(device)
elif cfg.test_data_type == "subway":
    dataset = get_subway_data(device)
elif cfg.test_data_type == "mcd":
    dataset = get_mcd_data(device)
else:
    raise ValueError(f"Unknown dataset name {cfg.train_dataset_type}!")


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

for index, (prob_name, points) in tqdm(
    enumerate(dataset), total=len(dataset), position=0
):
    prev_time = time.time()
    cluster_centers, selected_indices = greedy_facility_location(
        points, cfg.test_num_centers, distance=cfg.distance_metric, device=points.device
    )
    objective = compute_objective(points, cluster_centers, cfg.distance_metric).item()
    print(prob_name, objective, time.time() - prev_time)

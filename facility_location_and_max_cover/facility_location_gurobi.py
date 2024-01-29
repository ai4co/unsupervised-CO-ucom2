from src.facility_location_methods import gurobi_facility_location
import time
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
elif cfg.test_data_type == "mcd":
    dataset = get_mcd_data(device)
elif cfg.test_data_type == "subway":
    dataset = get_subway_data(device)
else:
    raise ValueError(f"Unknown dataset name {cfg.train_dataset_type}!")

time_budget = 120
try:
    i_seed = int(args.timestamp)
except:
    i_seed = time.time_ns()

for index, (prob_name, points) in enumerate(dataset):
    prev_time = time.time()
    ip_obj, ip_scores = gurobi_facility_location(
        points,
        cfg.test_num_centers,
        distance=cfg.distance_metric,
        linear_relaxation=False,
        timeout_sec=time_budget,
        verbose=cfg.verbose,
        seed=i_seed,
    )
    print(prob_name, ip_obj, time.time() - prev_time)

import sys

sys.path.append(2 * "../")

from typing import Optional
from einops import rearrange
from matplotlib.axes import Axes

import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.models.nn.utils import rollout, random_policy
from rl4co.models.zoo import AttentionModel, AutoregressivePolicy
from rl4co.utils.trainer import RL4COTrainer

# prepare hyperparameters and datasets
import pickle
import argparse

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=3)
    args = parser.parse_args()
except:
    class Argument:
        pass
    args = Argument()
    args.lr = 1e-4
    args.gpu = 3

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data: n_data, n_points, dim_point
# distances: n_data, n_points, n_points
data_train = torch.load(f"facility_location_rand800_train.pt", map_location=device)
locs_a = data_train.unsqueeze(-2)
locs_b = data_train.unsqueeze(-3)
distances_train = ((locs_a - locs_b) ** 2.0).sum(-1)

data_test = torch.load(f"facility_location_rand800_train.pt", map_location=device)
locs_a = data_test.unsqueeze(-2)
locs_b = data_test.unsqueeze(-3)
distances_test = ((locs_a - locs_b) ** 2.0).sum(-1)

n_points = 800
n_choose = 50
dim_point = 2
dim_embed = 128
max_dist = 2.0




# td train
data_ = data_train
n_data = len(data_)
locations_tensor = data_train
distances_tensor = distances_train

td_raw_train = TensorDict(
    {
        # given information; constant for each given instance                
        "locations": locations_tensor,  # (batch_size, n_points, dim_loc)                
        "orig_distances": distances_tensor,  # (batch_size, n_points, n_points)
        
        # the improvement in the total distance if the corresponding location is chosen
        "distances": max_dist - distances_tensor,  # (batch_size, n_points, n_points)
        
        # states changed by actions
        "chosen": torch.zeros(
            *locations_tensor.shape[:-1], dtype=torch.bool, device=device
        ),  # each entry is binary; 1 iff the corresponding facility is chosen
        "i": torch.zeros(
            n_data, dtype=torch.int64, device=device
        ),  # the number of points we have chosen
    },
    batch_size=n_data,
).to(device)

# td test
data_ = data_test
n_data = len(data_)
locations_tensor = data_test
distances_tensor = distances_test

td_raw_test = TensorDict(
    {
        # given information; constant for each given instance                
        "locations": locations_tensor,  # (batch_size, n_points, dim_loc)                
        "orig_distances": distances_tensor,  # (batch_size, n_points, n_points)
        "distances": distances_tensor,  # (batch_size, n_points, n_points)        
        # states changed by actions
        "chosen": torch.zeros(
            *locations_tensor.shape[:-1], dtype=torch.bool, device=device
        ),  # each entry is binary; 1 iff the corresponding facility is chosen
        "i": torch.zeros(
            n_data, dtype=torch.int64, device=device
        ),  # the number of points we have chosen
    },
    batch_size=n_data,
).to(device)

from rl4co.data.dataset import TensorDictDataset

class FLEnv(RL4COEnvBase):
    """
    Facility Location Problem environment
    At each step, the agent chooses a location.
    The reward is (-) the total distance of each location to its closest chosen location.

    Args:
        num_loc: number of locations (facilities) in the FL
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
    """

    name = "FL"

    def __init__(
        self,
        n_points: int = 500,        
        n_choose: int = 50,        
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_points = n_points
        self.n_choose = n_choose        
        self._make_spec(td_params)
    
    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[int] = None
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td.batch_size

        # Data generation: if not provided, generate a new batch of data
        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        self.device = td.device

        td_reset = TensorDict(
            {
                # given information
                "locations": td["locations"],  # (batch_size, n_points, dim_loc)
                "orig_distances": td["orig_distances"],  # (batch_size, n_points, n_points)
                "distances": td["distances"],  # (batch_size, n_points, n_points)
                
                # states changed by actions
                "chosen": torch.zeros(
                    *td["locations"].shape[:-1], dtype=torch.bool, device=td.device
                ),  # each entry is binary; 1 iff the corresponding facility is chosen
                "i": torch.zeros(
                    *batch_size, dtype=torch.int64, device=td.device
                ),  # the number of sets we have chosen
            },
            batch_size=batch_size,
        )

        # Compute action mask: mask out actions that are not allowed
        # for facility location, it would be choosing a set that has been chosen
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _step(self, td: TensorDict) -> TensorDict:
        # action: [batch_size, facility ID to be chosen]
        selected = td["action"]
        batch_size = selected.shape[0]

        # ====  Facility selction status update ====
        # (batch_size, n_sets)
        chosen = td["chosen"].clone()
        n_points_ = chosen.shape[-1]
        chosen[torch.arange(batch_size).to(td.device), selected] = True

        # finish if choosing enough facilities
        done = td["i"] >= (self.n_choose - 1)
        # the reward is calculated outside via get_reward for efficiency, so we set it to -inf here
        reward = torch.ones_like(done) * float("-inf")

        # update distances
        orig_distances = td["orig_distances"]
        # (batch_size, n_points)
        cur_min_dist = gather_by_index(orig_distances, chosen.nonzero(as_tuple=True)[1].reshape(batch_size, -1)).reshape(batch_size, -1, n_points_).min(dim=1).values.unsqueeze(-2)        
        dist_improve = torch.maximum(cur_min_dist - orig_distances, torch.tensor(0))
        
        # Return
        td_step = TensorDict(
            {
                "next": {
                    # given information
                    "locations": td["locations"],  # (batch_size, n_points, dim_loc)
                    "orig_distances": td[
                        "orig_distances"
                    ],  # (batch_size, n_points, n_points)
                    "distances": dist_improve,  # (batch_size, n_points, n_points)
                    # states changed by actions
                    "chosen": chosen,  # each entry is binary; 1 iff the corresponding facility is chosen
                    "i": td["i"] + 1,  # the number of sets we have chosen
                    "done": done,
                    "reward": reward,
                },
            },
            batch_size=td.batch_size,
        )
        td_step["next"].set("action_mask", self.get_action_mask(td_step["next"]))
        return td_step
    
    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        # we cannot choose the already-chosen facilities
        return ~td["chosen"]

    def get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:                
        chosen = td["chosen"]
        batch_size = td["chosen"].shape[0]
        n_points_ = td["chosen"].shape[-1]
        orig_distances = td["orig_distances"]
        cur_min_dist = gather_by_index(orig_distances, chosen.nonzero(as_tuple=True)[1].reshape(batch_size, -1)).reshape(batch_size, -1, n_points_).min(1).values.sum(-1)
        return -cur_min_dist
    
    def _make_spec(self, td_params: TensorDict = None):
        pass
    
    def generate_data(self, batch_size) -> TensorDict:
        pass

    def dataset(self, *args, **kwargs):
        return TensorDictDataset(td_raw_test)
        # return TensorDictDataset(td_raw_list[self.i_data])
        if kwargs.get("phase", None) == "train":
            return TensorDictDataset(td_raw_train)
        else:
            return TensorDictDataset(td_raw_test)
        
class MCInitEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.relu = nn.ReLU()       
        self.init_embed_distance1 = nn.Linear(n_points, embedding_dim)        
        self.init_embed_distance2 = nn.Linear(embedding_dim, embedding_dim)        
        self.init_embed_distance3 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, td: TensorDict):
        distances = td["distances"]
        distance_emb = self.init_embed_distance1(distances)
        distance_emb = self.relu(distance_emb)
        distance_emb = self.init_embed_distance2(distance_emb)
        distance_emb = self.relu(distance_emb)
        distance_emb = self.init_embed_distance3(distance_emb)               
        return distance_emb
    
class MCInitEmbedding2(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.init_embed_location1 = nn.Linear(dim_point, embedding_dim // 2)
        self.init_embed_location2 = nn.Linear(embedding_dim // 2, embedding_dim // 2)
        self.init_embed_location3 = nn.Linear(embedding_dim // 2, embedding_dim // 2)
                                
        self.init_embed_distance1 = nn.Linear(n_points, embedding_dim // 2)        
        self.init_embed_distance2 = nn.Linear(embedding_dim // 2, embedding_dim // 2)        
        self.init_embed_distance3 = nn.Linear(embedding_dim // 2, embedding_dim // 2)

    def forward(self, td: TensorDict):
        locations = td["locations"]
        distances = td["distances"]        
        
        location_emb = self.init_embed_location1(locations)
        location_emb = self.relu(location_emb)
        location_emb = self.init_embed_location2(location_emb)
        location_emb = self.relu(location_emb)
        location_emb = self.init_embed_location3(location_emb)
        
        distance_emb = self.init_embed_distance1(distances)
        distance_emb = self.relu(distance_emb)
        distance_emb = self.init_embed_distance2(distance_emb)
        distance_emb = self.relu(distance_emb)
        distance_emb = self.init_embed_distance3(distance_emb)
               
        return torch.cat((location_emb, distance_emb), dim=-1)
    

class MCContextEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, step_context_dim=None, linear_bias=True):
        super(MCContextEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        # self.W_placeholder = nn.Parameter(torch.Tensor(self.embedding_dim).uniform_(-1, 1))
        self.project_context = nn.Linear(embedding_dim, embedding_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        # max pooling of the embeddings of the chosen locations
        # embeddings: (batch_size, n_sets, embedding_dim)
        chosen_sets = td["chosen"]  # (batch_size, n_sets)
        # return: (batch_size, embedding_dim)
        return (embeddings * chosen_sets.unsqueeze(-1)).max(-2).values        

    def forward(self, embeddings, td):
        if td["i"][0] == 0:
            return torch.zeros(*td.batch_size, self.embedding_dim).to(td.device)
        # cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        cur_node_embedding = self._cur_node_embedding(embeddings, td)        
        return self.project_context(cur_node_embedding)
        # return torch.zeros_like(cur_node_embedding)


class StaticEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0
   


# Instantiate our environment

import time
time_s = time.time()

env = FLEnv(n_points=n_points, n_choose=n_choose)

# Instantiate policy with the embeddings we created above
policy = AutoregressivePolicy(
    env,
    embedding_dim=dim_embed,
    init_embedding=MCInitEmbedding2(dim_embed),
    context_embedding=MCContextEmbedding(dim_embed),
    dynamic_embedding=StaticEmbedding(dim_embed),
)

# Model: default is AM with REINFORCE and greedy rollout baseline
model = AttentionModel(
    env,
    policy=policy,
    baseline="rollout",
    batch_size=50,
    train_data_size=100,
    val_data_size=100,
    # optimizer_kwargs={"lr": 1e-6},
    optimizer_kwargs={"lr": args.lr},
)
# Greedy rollouts over untrained model


# We use our own wrapper around Lightning's `Trainer` to make it easier to use
# trainer = RL4COTrainer(max_epochs=100_000)
trainer = RL4COTrainer(max_epochs=100_000)
trainer.fit(model)

# test

data_ = data_test
n_data = data_.shape[0]
locations_tensor = data_test
distances_tensor = distances_test

td_test_list = []

for i_data in range(n_data):    
    locations_tensor_i = locations_tensor[i_data].unsqueeze(0)
    distances_tensor_i = distances_tensor[i_data].unsqueeze(0)
    td_test_ = TensorDict(
        {
            # given information; constant for each given instance                
            "locations": locations_tensor_i,  # (batch_size, n_points, dim_loc)                
            "orig_distances": distances_tensor_i,  # (batch_size, n_points, n_points)
            "distances": distances_tensor_i,  # (batch_size, n_points, n_points)        
            # states changed by actions
            "chosen": torch.zeros(
                *locations_tensor_i.shape[:-1], dtype=torch.bool, device=device
            ),  # each entry is binary; 1 iff the corresponding facility is chosen
            "i": torch.zeros(
                1, dtype=torch.int64, device=device
            ),  # the number of points we have chosen
        },
        batch_size=1,
    ).to(device)
    td_test_list.append(td_test_)


rewards = []

for td_ in td_test_list:
    td_init = env.reset(tensordict=td_).to(device)
    model = model.to(device)
    out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
    reward_final = out["reward"].mean().item()
    rewards.append(reward_final)

import numpy as np

time_e = time.time()
time_used = time_e - time_s
avg_obj = np.mean(rewards)
print(f"used time = {time_used:.3f}, avg obj = {avg_obj:3f}")
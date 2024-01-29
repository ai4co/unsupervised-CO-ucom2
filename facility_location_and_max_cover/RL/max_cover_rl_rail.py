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

import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--gpu", type=int, default=3)
parser.add_argument("--ds", type=str, required=True)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare hyperparameters and datasets
import pickle

with open(f"max_covering_rail_{args.ds}.pkl", "rb") as f:
    data_test = pickle.load(f)
with open(f"max_covering_rail_{args.ds}.pkl", "rb") as f:
    data_train = pickle.load(f)

ds_name, weights, membership = data_test[0]

n_sets = len(membership)
n_choose = 50
n_items = len(weights)
dim_embed = 128

# td train
data_ = data_train
n_data = len(data_)
membership_tensor = torch.zeros(n_data, n_sets, n_items).to(device)
weights_tensor = torch.zeros(n_data, n_items).to(device)

for i_data, (_, weights, membership) in enumerate(data_):
    for i_set, items_i in enumerate(membership):
        membership_tensor[i_data, i_set, items_i] = 1
    weights_tensor[i_data] += torch.tensor(weights).to(device)

td_raw_train = TensorDict(
    {
        # given information; constant for each given instance
        "orig_membership": membership_tensor,  # (batch_size, n_sets, n_items)
        "membership": membership_tensor,  # (batch_size, n_sets, n_items)
        "orig_weights": weights_tensor,  # (batch_size, n_items)
        "weights": weights_tensor,  # (batch_size, n_items)
        # states changed by actions
        "chosen": torch.zeros(
            *membership_tensor.shape[:-1], dtype=torch.bool, device=device
        ),  # each entry is binary; 1 iff the corresponding facility is chosen
        "i": torch.zeros(
            n_data, dtype=torch.int64, device=device
        ),  # the number of sets we have chosen
    },
    batch_size=n_data,
).to(device)

# td test
data_ = data_test
n_data = len(data_)
membership_tensor = torch.zeros(n_data, n_sets, n_items).to(device)
weights_tensor = torch.zeros(n_data, n_items).to(device)

for i_data, (_, weights, membership) in enumerate(data_):
    for i_set, items_i in enumerate(membership):
        membership_tensor[i_data, i_set, items_i] = 1
    weights_tensor[i_data] += torch.tensor(weights).to(device)

td_raw_test = TensorDict(
    {
        # given information; constant for each given instance
        "orig_membership": membership_tensor,  # (batch_size, n_sets, n_items)
        "membership": membership_tensor,  # (batch_size, n_sets, n_items)
        "orig_weights": weights_tensor,  # (batch_size, n_items)
        "weights": weights_tensor,  # (batch_size, n_items)
        # states changed by actions
        "chosen": torch.zeros(
            *membership_tensor.shape[:-1], dtype=torch.bool, device=device
        ),  # each entry is binary; 1 iff the corresponding facility is chosen
        "i": torch.zeros(
            n_data, dtype=torch.int64, device=device
        ),  # the number of sets we have chosen
    },
    batch_size=n_data,
).to(device)

from rl4co.data.dataset import TensorDictDataset


class MCEnv(RL4COEnvBase):
    """
    Max Covering Problem environment
    At each step, the agent chooses a set.
    The reward is the total weights of the covered items.

    Args:
        num_loc: number of locations (facilities) in the FL
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
    """

    name = "MC"

    def __init__(
        self,
        n_set: int = 50,
        n_items: int = 100,
        n_sets_to_choose: int = 5,
        i_data: int = 0,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_set = n_set
        self.n_items = n_items
        self.n_sets_to_choose = n_sets_to_choose
        self.i_data = i_data
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
                # given information; constant for each given instance
                "orig_membership": td["orig_membership"],  # (batch_size, n_sets, n_items)
                "membership": td["membership"],  # (batch_size, n_sets, n_items)
                "orig_weights": td["orig_weights"],  # (batch_size, n_items)
                "weights": td["weights"],  # (batch_size, n_items)
                # states changed by actions
                "chosen": torch.zeros(
                    *td["membership"].shape[:-1], dtype=torch.bool, device=td.device
                ),  # each entry is binary; 1 iff the corresponding facility is chosen
                "i": torch.zeros(
                    *batch_size, dtype=torch.int64, device=td.device
                ),  # the number of sets we have chosen
            },
            batch_size=batch_size,
        )

        # Compute action mask: mask out actions that are not allowed
        # for max covering, it would be choosing a set that has been chosen
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _step(self, td: TensorDict) -> TensorDict:
        # action: [batch_size, facility ID to be chosen]
        batch_size = td["action"].shape[0]
        selected = td["action"]

        # ====  Facility slection status update ====
        # (batch_size, n_sets)
        chosen = td["chosen"].clone()
        chosen[torch.arange(batch_size).to(td.device), selected] = True

        # finish if all choose enough facilities
        done = td["i"] >= (self.n_sets_to_choose - 1)
        # The reward is calculated outside via get_reward for efficiency, so we set it to -inf here
        reward = torch.ones_like(done) * float("-inf")

        remaining_sets = ~td["chosen"]  # (batch_size, n_sets)
        covered_items = (
            (td["chosen"].unsqueeze(-1).float() * td["orig_membership"])
            .max(dim=-2)
            .values
        )  # (batch_size, n_itmes)
        remaining_items = 1.0 - covered_items  # (batch_size, n_itmes)

        # Return
        td_step = TensorDict(
            {
                "next": {
                    "orig_membership": td["orig_membership"],
                    "membership": (remaining_sets.unsqueeze(-1)) * td["membership"],
                    "orig_weights": td["orig_weights"],
                    "weights": td["weights"] * remaining_items,
                    "chosen": chosen,
                    "i": td["i"] + 1,
                    "done": done,
                    "reward": reward,
                },
            },
            # td.shape,
            batch_size=td.batch_size,
        )
        td_step["next"].set("action_mask", self.get_action_mask(td_step["next"]))
        return td_step

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        # we cannot choose the already-chosen facilities
        return ~td["chosen"]

    def get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        membership = td["orig_membership"]  # (batch_size, n_sets, n_items)
        weights = td["orig_weights"]  # (batch_size, n_itmes)
        chosen_sets = td["chosen"]  # (batch_size, n_set)
        chosen_items = (
            (chosen_sets.unsqueeze(-1).float() * membership).max(dim=-2).values
        )  # (batch_size, n_itmes)
        chosen_weights = (chosen_items * weights).sum(dim=-1)
        return chosen_weights

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
        self.init_embed_membership = nn.Linear(n_items, embedding_dim)
        self.relu = nn.ReLU()
        self.init_embed_membership2 = nn.Linear(embedding_dim, embedding_dim)
        self.init_embed_membership3 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, td: TensorDict):
        membership_weighted = td["membership"] * td["weights"].unsqueeze(-2)
        membership_emb = self.init_embed_membership(membership_weighted)
        membership_emb = self.relu(membership_emb)
        membership_emb = self.init_embed_membership2(membership_emb)
        membership_emb = self.relu(membership_emb)
        membership_emb = self.init_embed_membership3(membership_emb)
        return membership_emb


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
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        return self.project_context(cur_node_embedding)


class StaticEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0


# Instantiate our environment

import time

time_s = time.time()

env = MCEnv(n_sets, n_items, n_choose)

# Instantiate policy with the embeddings we created above
policy = AutoregressivePolicy(
    env,
    embedding_dim=dim_embed,
    init_embedding=MCInitEmbedding(dim_embed),
    context_embedding=MCContextEmbedding(dim_embed),
    dynamic_embedding=StaticEmbedding(dim_embed),
)

# Model: default is AM with REINFORCE and greedy rollout baseline
model = AttentionModel(
    env,
    policy=policy,
    baseline="rollout",
    batch_size=1,
    train_data_size=1,
    val_data_size=1,
    optimizer_kwargs={"lr": args.lr},
)
# Greedy rollouts over untrained model


# We use our own wrapper around Lightning's `Trainer` to make it easier to use
trainer = RL4COTrainer(max_epochs=1_000)
trainer.fit(model)

td_init = env.reset(tensordict=td_raw_test).to(device)
model = model.to(device)
out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
reward_final = out["reward"].mean().item()

import numpy as np

time_e = time.time()
time_used = time_e - time_s
avg_obj = reward_final
print(f"used time = {time_used:.3f}, avg obj = {avg_obj:3f}")

Note: built up based on https://github.com/Thinklab-SJTU/One-Shot-Cardinality-NN-Solver

We use the following packages
```
torch                  1.11.0
torch-geometric        2.1.0
torch-scatter          2.0.9
torch-sparse           0.6.13
torch-spline-conv      1.2.1
```

First, install the required packages
```
pip install torch==1.11.0+${CUDA} --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-sparse==0.6.13 torch-spline-conv==1.2.1 torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
pip install torch-geometric==2.1.0.post1
pip install ortools easydict matplotlib yfinance pandas cvxpylayers xlwt pyyaml gurobipy
```

We have the following methods
- CPU methods           : random / gurobi / scip / greedy
- CardNN methods        : cardnn-s / cardnn-gs / cardnn-hgs 
- CardNN-noTTO methods  : cardnn-notto-s / cardnn-notto-gs / cardnn-notto-hgs
- EGN method            : egn-naive
- Ours (UCom2)          : ucom2

## Facility location
Below are the datasets for the facility location problem
- Datasets               : rand500 / rand800 / starbucks / mcd / subway

Below are the commands for running CPU methods
- CPU methods           : random / gurobi / scip / greedy

```
python facility_location_[method].py --cfg cfg/facility_location_[dataset].yaml --timestamp [seed]
```
For example, to run the method "random" on "rand500" datasets, you should run
```
python facility_location_random.py --cfg cfg/facility_location_rand500.yaml --timestamp 0
```

Below are the commands for running CardNN and CardNN-noTTO methods
- CardNN methods        : cardnn-s / cardnn-gs / cardnn-hgs
- CardNN-noTTO methods  : cardnn-notto-s / cardnn-notto-gs / cardnn-notto-hgs

```
python facility_location_[method].py --cfg cfg/facility_location_[dataset].yaml --lr [learning_rate] --reg [gumbel_sigma] --timestamp [seed]
```

For example, to run the method "cardnn-hgs" on "rand500" datasets, you should run
```
python facility_location_cardnn-hgs.py --cfg cfg/facility_location_rand500.yaml --lr 1e-4 --reg 0.25 --timestamp 0
```

Below are the commands for running EGN method and our method
- EGN method            : egn-naive
- Ours (UCom2)          : ucom2

```
python facility_location_[method].py --cfg cfg/facility_location_[dataset].yaml --lr [learning_rate] --reg [constraint_coefficient] --timestamp [seed]
```

For example, to run the proposed method "ucom2" on "rand500" datasets, you should run
```
python facility_location_ucom2.py --cfg cfg/facility_location_rand500.yaml --lr 1e-2 --reg 1e-2 --timestamp 0
```

For the RL method, we adapted code from https://github.com/ai4co/rl4co.
Please follow the installation guidance thereof.

See the files in the RL folder: facility_location_rl_*.py.

For example, to run the RL method on "rand500" datasets, you should run
``` 
python RL/facility_location_rl_rand500.py --ds rand500_train
```

## Maximum coverage
Below are the datasets for the maximum coverage problem
- Datasets               : rand500 / rand1000 / twitch / rail

Below are the commands for running CPU methods
- CPU methods           : random / gurobi / scip / greedy

```
python max_cover_[method].py --cfg cfg/max_cover_[dataset].yaml --timestamp [seed]
```

For example, to run the method "random" on "rand500" datasets, you should run
```
python max_cover_random.py --cfg cfg/max_cover_rand500.yaml --timestamp 0
```

Below are the command for running CardNN and CardNN-noTTO methods
- CardNN methods        : cardnn-s / cardnn-gs / cardnn-hgs 
- CardNN-noTTO methods  : cardnn-notto-s / cardnn-notto-gs / cardnn-notto-hgs

```
python max_cover_[method].py --cfg cfg/max_cover_[dataset].yaml --lr [learning_rate] --reg [gumbel_sigma] --timestamp [seed]
```

For example, to run the method "cardnn-hgs" on "rand500" datasets, you should run
```
python max_cover_cardnn-hgs.py --cfg cfg/max_cover_rand500.yaml --lr 1e-5 --reg 0.15 --timestamp 0
```

Below are the command for running EGN method and our method
- EGN method            : egn-naive
- ours (UCom2)          : ucom2

```
python max_cover_[method].py --cfg cfg/max_cover_[dataset].yaml --lr [learning_rate] --reg [constraint_coefficient] --timestamp [seed]
```

For example, to run the proposed method "ucom2" on "rand500" datasets, you should run
```
python max_cover_ucom2.py --cfg cfg/max_cover_rand500.yaml --lr 1e-5 --reg 0.15 --timestamp 0
```

For the RL method, we adapted code from https://github.com/ai4co/rl4co.
Please follow the installation guidance thereof.

See the files in the RL folder: max_cover_rl_*.py.

For example, to run the RL method on "rand500" datasets, you should run
```
python RL/max_cover_rl_rand500.py --ds rand500_train
```

## Batch files

We also provide bash files for conveniently executing experiments for each method.

Below are the commands for running CPU methods
- CPU methods           : random / gurobi / scip / greedy

```
bash [method].sh
```

For example, to execute experiments with the method "random", you should run
```
bash random.sh
```

For the other methods, two bash files are provided for two problems (facility location and maximum coverage), perspectively.

```
bash [fl/mc]_{method}.sh
```

For example, to execute experiments on the facility location problem with the proposed method "ucom2", you should run
```
bash fl_ours.sh
```
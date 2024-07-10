We use the following packages
```
torch                  1.11.0
geneticalgorithm       1.0.2
```

First, install the required packages
```
pip install torch==1.11.0+${CUDA} --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install geneticalgorithm
```

We have the following methods
- Baseline methods      : greedy-RD / greedy-GA / DC / gurobi
- Ours (UCom2)          : ucom2 / ucom2-cpu

Below are the datasets for the robust coloring problem:
- Datasets               : collins / gavin / krogan / ppi

Below are the commands for running the baseline methods

```
python greedy-RD.py --ds [dataset] --c [number_of_colors]
python greedy-GA.py --ds [dataset] --c [number_of_colors]
python DC.py --ds [dataset] --c [number_of_colors]
python gurobi.py --graph_name [dataset] --num_colors [number_of_colors] --timeout_sec 300 --num_seeds 5
```

Below are the commands for running the proposed method
```
python ucom2.py --ds [dataset] --c [number_of_colors] --npam [number_of_initial_parameters]
python ucom2-cpu.py --ds [dataset] --c [number_of_colors] --npam [number_of_initial_parameters]
```

Note: With a larger number of initial parameters, you use more time and memory for better optimization performance

import gurobipy as grb
import numpy as np
import argparse
import pickle
import time
import os

import pdb


def metric(assign, A_soft):
    return ((assign @ assign.T) * A_soft.numpy()).sum()

def gurobi_coloring(root, graph_name, graph_hard, graph_soft, num_nodes, num_colors, timeout_sec, seed): 
    save_path = os.path.join(root, 'trained_model', 'gurobi', f'{graph_name}_nc{num_colors}_to{timeout_sec}_{seed}.mst')
    
    model = grb.Model('robust graph coloring')
    model.setParam('LogToConsole', 1)
    model.setParam('TimeLimit', timeout_sec)
    model.setParam('Seed', seed)

    var_x = {}
    var_y = {}

    for node in range(num_nodes):
        var_x[node] = {}
        var_y[node] = {}
        for color in range(num_colors): 
            var_x[node][color] = model.addVar(vtype=grb.GRB.BINARY, name=f'x_{node}_{color}')
        if node in graph_soft:
            for soft_neigh in graph_soft[node]: 
                var_y[node][soft_neigh] = model.addVar(vtype=grb.GRB.BINARY, name=f'y_{node}_{soft_neigh}')
        
    const_1 = {}
    const_2 = {}
    const_3 = {}
    for node in range(num_nodes):
        const_1[node] = 0
        for color in range(num_colors): 
            const_1[node] += var_x[node][color]
        model.addConstr(const_1[node] == 1) 
        
        const_2[node] = {}
        const_3[node] = {}
        if node in graph_hard:
            for neigh in graph_hard[node]:
                const_2[node][neigh] = {}
                
                for color in range(num_colors):
                    const_2[node][neigh][color] = var_x[node][color] + var_x[neigh][color]
                    model.addConstr(const_2[node][neigh][color] <= 1) 
        if node in graph_soft: 
            for soft_neigh in graph_soft[node]:
                const_3[node][soft_neigh] = {}
                for color in range(num_colors):
                    const_3[node][soft_neigh][color] = var_x[node][color] + var_x[soft_neigh][color] - 1
                    model.addConstr(const_3[node][soft_neigh][color] <= var_y[node][soft_neigh])

    penalty = 0
    for node in range(num_nodes):
        if node in graph_soft:
            for soft_neigh in graph_soft[node]:
                penalty += graph_soft[node][soft_neigh] * var_y[node][soft_neigh]
    model.setObjective(penalty, grb.GRB.MINIMIZE)
    
    model.optimize()
    model.write(save_path)
    
    assign = np.zeros([num_nodes, num_colors])
    for n in range(num_nodes):
        for c in range(num_colors):
            try:
                assign[n][c] = model.getVarByName(f'x_{n}_{c}').X
            except:
                pdb.set_trace()
                
    return assign
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robust Graph Coloring')
    parser.add_argument('--root', type=str, default=os.path.join(os.curdir, 'data', 'gurobi'))
    parser.add_argument('--graph_name', type=str, default='krogan')
    parser.add_argument('--num_colors', type=int, default=10)
    parser.add_argument('--timeout_sec', type=int, default=100)
    parser.add_argument('--num_seeds', type=int, default=5)
    
    args = parser.parse_args()
    # ['krogan', 'collins', 'gavin']
    
    with open(os.path.join(args.root, f'{args.graph_name}_new.pkl'), 'rb') as f:
        A_soft, A_hard = pickle.load(f)
        
    edge_soft = A_soft.nonzero()
    edge_hard = A_hard.nonzero()

    edge_soft = edge_soft[edge_soft[:, 0] < edge_soft[:, 1]]
    edge_hard = edge_hard[edge_hard[:, 0] < edge_hard[:, 1]]
    
    graph_soft, graph_hard = dict(), dict()
    
    for src, dst in edge_soft:
        if src.item() not in graph_soft: graph_soft[src.item()] = dict()
        graph_soft[src.item()][dst.item()] = A_soft[src.item()][dst.item()].item()
        
    for src, dst in edge_hard:
        if src.item() not in graph_hard: graph_hard[src.item()] = list()
        graph_hard[src.item()].append(dst.item())

    num_nodes = A_soft.shape[0]
    num_colors = args.num_colors
    timeout_sec = args.timeout_sec
    avg_loss = 0
    avg_execute_time = 0
    for seed in range(args.num_seeds):
        start = time.time()
        assign = gurobi_coloring(args.root, args.graph_name, graph_hard, graph_soft, num_nodes, num_colors, timeout_sec, seed)
        end = time.time()
        loss = metric(assign, A_soft)
        excute_time = end - start
        print(f'SEED: {seed}   Objective: {loss:.4f}   Time: {excute_time:.4f}')
        avg_loss += loss
        avg_execute_time += excute_time
    avg_loss = avg_loss / args.num_seeds
    avg_execute_time = avg_execute_time / args.num_seeds
    print(f'FINAL Objective: {avg_loss:.4f}   Time: {avg_execute_time:.4f}')

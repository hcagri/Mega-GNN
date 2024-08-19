import pandas as pd 
import numpy as np 
import torch 

import timeit
import ports_cpp # type: ignore

import ports as cython_ports #type: ignore
import multiprocessing as mp


def process_ports(df, direction = ['source', 'target']):
    df = df.sort_values([direction[1], 't'])
    n_edges = df.shape[0]
    ports = np.zeros((n_edges, ), dtype=np.int32)
    array = df[direction].to_numpy(dtype=np.int32)
    ports = cython_ports.assign_ports(array, ports)
    return torch.tensor(ports)[np.argsort(df.index)]

def assign_ports_batch(edge_index, timestamp):

    df = pd.DataFrame(torch.cat([edge_index.T, timestamp.reshape((-1,1))], dim=1).numpy().astype('int'), columns=['source', 'target', 't'])

    with mp.Pool(2) as pool:
        ports_1, ports_2 = pool.starmap(process_ports, [
            (df, ['source', 'target']),
            (df, ['target', 'source'])
        ])
    
    ports = torch.stack([ports_1, ports_2], dim=1)
    return ports

def to_adj_nodes_with_times(num_nodes: int, edges: np.array):
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u,v,t in edges:
        u,v,t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out

def ports(edge_index, adj_list):
    ports_arr = np.zeros((edge_index.shape[1], 1))
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1: continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:,[0]],return_index=True,axis=0)
        nbs_unique = a[np.sort(idx)][:,0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u,v)] = i
        
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.clone().numpy()
    for i, e in enumerate(edge_index.T):
        ports_arr[i] = ports_dict[tuple(e)]
    return ports_arr

def add_ports(num_nodes, edge_index, edges):
    '''Adds port numberings to the edge features'''
    adj_list_in, adj_list_out = to_adj_nodes_with_times(num_nodes, edges)
    in_ports = ports(edge_index, adj_list_in)
    out_ports = ports(edge_index.flipud(), adj_list_out)
    return in_ports, out_ports


# Method 1
def method1():
    ports_1, ports_2 = ports_cpp.assign_ports(edges, edge_index.numpy().astype('int'), graph.num_nodes)
    ports_from_cpp = torch.stack([torch.tensor(ports_1), torch.tensor(ports_2)], dim=1)
    return ports_from_cpp

# Method 2
def method2():
    ports_1, ports_2 = add_ports(graph.num_nodes, edge_index, edges)
    ports_from_python = torch.stack([torch.tensor(ports_1), torch.tensor(ports_2)], dim=1)
    return ports_from_python

# Method 3
def method3():
    ports_from_cython = assign_ports_batch(edge_index, timestamp)
    return ports_from_cython



def benchmark():
    num_runs = 5  # Number of times to run each method to get a reliable average
    
    time1 = timeit.timeit(method1, number=num_runs)
    # time2 = timeit.timeit(method2, number=num_runs)
    # time3 = timeit.timeit(method3, number=num_runs)

    print(f"{'C++ average time:':30} {time1 / num_runs:.6f} seconds")
    # print(f"{'Pure Python average time:':30} {time2 / num_runs:.6f} seconds")
    # print(f"{'Python + Cython average time:':30} {time3 / num_runs:.6f} seconds")

if __name__ == "__main__":
    graph = torch.load('sample_data/batch.pth')

    edge_index = torch.cat([graph['node', 'to', 'node'].edge_index, graph['node', 'rev_to', 'node'].edge_index], dim=1)
    timestamp = torch.cat([graph.timestamps, torch.zeros((graph['node', 'rev_to', 'node'].num_edges,))], dim=0)
    edges = torch.cat([edge_index.T, timestamp.reshape((-1,1))], dim=1).numpy().astype('int')
    

    benchmark()
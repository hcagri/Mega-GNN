import torch
import torch.nn as nn
from genagg import GenAgg
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import coalesce
from torch_geometric.transforms import RemoveDuplicatedEdges
from torch_scatter import scatter
import time
from genagg import GenAgg


class MultiEdgeAggModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.agg = GenAgg()
        
    def forward(self, edge_index, edge_attr, simp_edge_batch):
        _, inverse_indices = torch.unique(simp_edge_batch, return_inverse=True)
        new_edge_index = scatter(edge_index, inverse_indices, dim=1, reduce='mean')
        new_edge_attr = self.agg(x=edge_attr, index=inverse_indices)
        return new_edge_index, new_edge_attr
    
    def reset_parameters(self):
        self.agg.reset_parameters()


model = MultiEdgeAggModule()
if hasattr(model, 'reset_parameters'):
    model.reset_parameters()


# new_edge_index, new_edge_attr = edge_model(batch['node', 'to', 'node'].edge_index, batch['node', 'to', 'node'].edge_attr, batch['node', 'to', 'node'].simp_edge_batch)




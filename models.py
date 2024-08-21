import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv
from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
import torch
import logging
import numpy as np
from torch_scatter import scatter
from torch_geometric.utils import degree
from genagg import GenAgg
from genagg.MLPAutoencoder import MLPAutoencoder
import math 


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerAgg(nn.Module):
    def __init__(self, d_model = 66):
        super().__init__()

        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=0.05, max_len=128)
        self.trans_enc = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=2, 
            dim_feedforward=128, 
            batch_first=True, # If True, then the input and output tensors are provided as (batch, seq, feature). 
            norm_first=True
            )

    def forward(self, x, index, timestamps):
        timestamps[timestamps == 0] = 0.001  #just to make it larger than 0
        # Add timestamps to the edge features to be able to sort according to them
        x = torch.cat([timestamps.view(-1, 1), x], dim=1)
        
        sort_ids = torch.argsort(index)
        dense_edge_feats, mask = to_dense_batch(x[sort_ids, :], index[sort_ids])
        sorted_dense_edge_feats, sorted_mask = self.sort_wrt_time(dense_edge_feats, mask)

        sorted_dense_edge_feats = self.pos_enc(sorted_dense_edge_feats.permute(1,0,2)).permute(1,0,2)
        sorted_dense_edge_feats = self.trans_enc(sorted_dense_edge_feats, src_key_padding_mask = sorted_mask)

        return sorted_dense_edge_feats.mean(dim=1).squeeze()

    def sort_wrt_time(self, matt, mask):
        first_feature = matt[:, :, 0] 
        sort_indices = torch.argsort(first_feature, dim=1)
        sorted_matt = torch.gather(matt, 1, sort_indices.unsqueeze(-1).expand(-1, -1, matt.shape[-1]))
        sorted_mask = torch.gather(mask, 1, sort_indices)
        return sorted_matt[:, :, 1:], sorted_mask

class GRUAgg(nn.Module):
    def __init__(self, d_model = 66):
        super().__init__()

        self.gru = nn.GRU(
            d_model, 
            hidden_size=d_model, 
            num_layers=2, 
            batch_first=True
            )

    def forward(self, x, index, timestamps):
        timestamps[timestamps == 0] = 0.001  #just to make it larger than 0
        # Add timestamps to the edge features to be able to sort according to them
        x = torch.cat([timestamps.view(-1, 1), x], dim=1)
        
        sort_ids = torch.argsort(index)
        dense_edge_feats, mask = to_dense_batch(x[sort_ids, :], index[sort_ids])
        sorted_dense_edge_feats, sorted_mask = self.sort_wrt_time(dense_edge_feats, mask)

        sorted_dense_edge_feats = self.gru(sorted_dense_edge_feats)[0]
        sorted_dense_edge_feats[~sorted_mask.unsqueeze(-1).expand(-1, -1, sorted_dense_edge_feats.shape[-1])] = 0

        return sorted_dense_edge_feats.mean(dim=1).squeeze()

    def sort_wrt_time(self, matt, mask):
        first_feature = matt[:, :, 0] 
        sort_indices = torch.argsort(first_feature, dim=1)
        sorted_matt = torch.gather(matt, 1, sort_indices.unsqueeze(-1).expand(-1, -1, matt.shape[-1]))
        sorted_mask = torch.gather(mask, 1, sort_indices)
        return sorted_matt[:, :, 1:], sorted_mask
    
    

class PnaAgg(nn.Module):
    def __init__(self , n_hidden, deg):
        super().__init__()
        
        aggregators = ['mean', 'min', 'max', 'std']
        self.num_aggregators = len(aggregators)
        scalers = ['identity', 'amplification', 'attenuation']

        self.agg = DegreeScalerAggregation(aggregators, scalers, deg)
        self.lin = nn.Linear(len(scalers)*len(aggregators)*n_hidden, n_hidden)

    def forward(self, x, index):
        out = self.agg(x, index)
        return self.lin(out)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                param = nn.init.kaiming_normal_(param.detach())
            elif 'bias' in name:
                param = nn.init.constant_(param.detach(), 0)
        


class IdentityAgg(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, index):
        return x

class GinAgg(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.nn = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(), 
                nn.Linear(n_hidden, n_hidden)
                )
    def forward(self, x, index):
        out = torch.relu(x)
        out = scatter(x, index, dim=0, reduce='sum')
        return self.nn(out)

class SumAgg(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, index):
        return scatter(x, index, dim=0, reduce='sum')

class MultiEdgeAggModule(nn.Module):
    def __init__(self, n_hidden=None, agg_type=None, index=None):
        super().__init__()
        self.agg_type = agg_type

        if agg_type == 'genagg':
            self.agg = GenAgg(f=MLPAutoencoder, jit=False)
        elif agg_type == 'gin':
            self.agg = GinAgg(n_hidden=n_hidden)
        elif agg_type == 'pna':
            d = degree(index, dtype=torch.long)
            deg = torch.bincount(d, minlength=1)[1:] # discard the value count for 0
            self.agg = PnaAgg(n_hidden=n_hidden, deg=deg)
        elif agg_type == 'sum':
            self.agg = SumAgg()
        elif agg_type == 'transformer':
            self.agg = TransformerAgg(d_model=n_hidden)
        elif agg_type == 'gru':
            self.agg = GRUAgg(d_model=n_hidden)
        else:
            self.agg = IdentityAgg()
        
    def forward(self, edge_index, edge_attr, simp_edge_batch, timestamps=None):
        _, inverse_indices = torch.unique(simp_edge_batch, return_inverse=True)
        new_edge_index = scatter(edge_index, inverse_indices, dim=1, reduce='mean') if self.agg_type is not None else edge_index
        new_edge_attr = self.agg(x=edge_attr, index=inverse_indices, timestamps=timestamps)
        return new_edge_index, new_edge_attr, inverse_indices
    
    def reset_parameters(self):
        self.agg.reset_parameters()

class GINe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_updates=False, residual=True, 
                edge_dim=None, dropout=0.0, final_dropout=0.5, flatten_edges=False, 
                edge_agg_type=None, node_agg_type=None, index_ = None, args=None):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.flatten_edges = flatten_edges
        self.args = args

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.edge_agg = MultiEdgeAggModule(n_hidden, agg_type=edge_agg_type, index=index_)

        if node_agg_type == 'genagg':
            self.node_agg = GenAgg(f=MLPAutoencoder, jit=False)
        elif node_agg_type == 'sum':
            self.node_agg = 'sum'

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden), 
                nn.ReLU(), 
                nn.Linear(self.n_hidden, self.n_hidden)
                ), 
                edge_dim=self.n_hidden, 
                aggr = self.node_agg # Added New!!!
                )
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        if args.data != 'ETH':
            self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                                Linear(25, n_classes))
        else:
            self.mlp = nn.Sequential(Linear(n_hidden, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                                Linear(25, n_classes))

    def forward(self, x, edge_index, edge_attr, simp_edge_batch=None, timestamps=None):

        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            if self.flatten_edges:
                n_edge_index, n_edge_attr, inverse_indices  = self.edge_agg(edge_index, edge_attr, simp_edge_batch, timestamps)
                x = (x + F.relu(self.batch_norms[i](self.convs[i](x, n_edge_index, n_edge_attr)))) / 2
                if self.edge_updates: 
                    remapped_edge_attr = torch.index_select(n_edge_attr, 0, inverse_indices) # artificall node attributes 
                    edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], remapped_edge_attr, edge_attr], dim=-1)) / 2
            else:
                x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
                if self.edge_updates: 
                    edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        if self.args.data != 'ETH':
            x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
            x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = x
        
        return self.mlp(out)
    
class GATe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, n_hidden=100, n_heads=4, edge_updates=False, edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        # GAT specific code
        tmp_out = n_hidden // n_heads
        n_hidden = tmp_out * n_heads

        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.dropout = dropout
        self.final_dropout = final_dropout
        
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)
        
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(self.num_gnn_layers):
            conv = GATConv(self.n_hidden, tmp_out, self.n_heads, concat = True, dropout = self.dropout, add_self_loops = True, edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden),nn.ReLU(),nn.Linear(self.n_hidden, self.n_hidden),))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))
                
        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(25, n_classes))
            
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        
        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
                    
        logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        logging.debug(f"x.shape = {x.shape}")
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        logging.debug(f"x.shape = {x.shape}")
        out = x

        return self.mlp(out)
    
class PNA(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_updates=True,
                edge_dim=None, dropout=0.0, final_dropout=0.5, deg=None):
        super().__init__()
        n_hidden = int((n_hidden // 5) * 5)
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=n_hidden, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                              Linear(25, n_classes))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates: 
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        logging.debug(f"x.shape = {x.shape}")
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        logging.debug(f"x.shape = {x.shape}")
        out = x
        return self.mlp(out)
    
class RGCN(nn.Module):
    def __init__(self, num_features, edge_dim, num_relations, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_update=False,
                residual=True,
                dropout=0.0, final_dropout=0.5, n_bases=-1):
        super(RGCN, self).__init__()

        self.num_features = num_features
        self.num_gnn_layers = num_gnn_layers
        self.n_hidden = n_hidden
        self.residual = residual
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.n_classes = n_classes
        self.edge_update = edge_update
        self.num_relations = num_relations
        self.n_bases = n_bases

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.mlp = nn.ModuleList()

        if self.edge_update:
            self.emlps = nn.ModuleList()
            self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
        
        for _ in range(self.num_gnn_layers):
            conv = RGCNConv(self.n_hidden, self.n_hidden, num_relations, num_bases=self.n_bases)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(self.n_hidden))

            if self.edge_update:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden),
                ))

        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                              Linear(25, n_classes))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, RGCNConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        edge_type = edge_attr[:, -1].long()
        #edge_attr = edge_attr[:, :-1]
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x =  (x + F.relu(self.bns[i](self.convs[i](x, edge_index, edge_type)))) / 2
            if self.edge_update:
                edge_attr = (edge_attr + F.relu(self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)))) / 2
        
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        x = self.mlp(x)
        out = x

        return x
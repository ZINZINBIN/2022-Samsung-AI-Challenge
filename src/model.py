import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, List
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv, GCNConv, GATv2Conv, GINEConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.nn.pool.graclus import graclus
from torch_scatter import scatter_max
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.nn import max_pool, max_pool_x, avg_pool
from torch_geometric.utils.undirected import to_undirected
from pytorch_model_summary import summary
from src.preprocessing import ATOMS_LIST, ATOMS_DEGREE, ATOMS_NUMHS, ATOMS_VALENCE, ATOMS_AROMATIC, ATOMS_RING, ATOMS_CHARGE, ATOMS_HYBRID, atom_properties

atom_feats = [
    len(ATOMS_LIST) + 1, 
    len(ATOMS_DEGREE) + 1, 
    len(ATOMS_NUMHS) + 1, 
    len(ATOMS_VALENCE) + 1, 
    len(ATOMS_AROMATIC),
    len(ATOMS_RING),
    len(ATOMS_CHARGE) + 1,
    len(ATOMS_HYBRID) + 1,
    # len(atom_properties)
]

class FeatureEmbedding(nn.Module):
    def __init__(self, feature_lens : List, max_norm = 1.0):
        super(FeatureEmbedding, self).__init__()
        self.feature_lens = feature_lens
        self.emb_layers = nn.ModuleList()
        self.max_norm = max_norm

        '''
        for size in feature_lens[:-1]:
            emb_layer = nn.Embedding(size, size, max_norm = max_norm)
            emb_layer.load_state_dict({'weight': torch.eye(size)})
            self.emb_layers.append(emb_layer)
        '''

        for size in feature_lens:
            emb_layer = nn.Embedding(size, size, max_norm = max_norm)
            emb_layer.load_state_dict({'weight': torch.eye(size)})
            self.emb_layers.append(emb_layer)

    def forward(self, x : torch.Tensor):
        output = []
        for i, layer in enumerate(self.emb_layers):
            output.append(layer(x[:, i].long()))

        # output.append(x[:, -self.feature_lens[-1]:])
        output = torch.cat(output, 1)
        return output

class GCNLayer(nn.Module):
    def __init__(self, in_features : int, out_features : int, alpha : float)->None:
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.gconv = GCNConv(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.act = nn.LeakyReLU(alpha)
        
    def forward(self, x, edge_idx = None):
        h = self.gconv(x, edge_idx)
        h = self.norm(h)
        h = self.act(h)
        return h

class GConvNet(nn.Module):
    def __init__(self, hidden, output_dim : int = 2, alpha = 0.01, embedd_max_norm = 1.0, n_layers : int = 4):
        super(GConvNet, self).__init__()
        torch.manual_seed(42)
        self.hidden = hidden
        self.alpha = alpha
        self.embedd_max_norm = embedd_max_norm
        self.output_dim = output_dim

        self.embedd = FeatureEmbedding(feature_lens = atom_feats, max_norm=embedd_max_norm)

        self.gc = nn.ModuleList()
        for i in range(n_layers):
            self.gc.append(GCNLayer(sum(atom_feats) if i == 0 else hidden, hidden, alpha))

        self.mlp = nn.ModuleList(
            [
                nn.Linear(self.hidden, self.hidden//2),
                nn.BatchNorm1d(self.hidden//2),
                nn.ReLU(),
                nn.Linear(self.hidden//2, self.hidden//2),
                nn.BatchNorm1d(self.hidden//2),
                nn.ReLU(),
                nn.Linear(self.hidden//2, output_dim),
            ]
        )
       
    def forward(self, inputs)->torch.Tensor:
        x, edge_idx, batch_idx = inputs.x,  inputs.edge_index, inputs.batch
        x = self.embedd(x)

        for layer in self.gc:
            x = layer(x, edge_idx)
        x = global_add_pool(x, batch_idx)

        for layer in self.mlp:
            x = layer(x)

        return x if self.output_dim != 1 else x.squeeze(1)

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))

class GATLayer(nn.Module):
    def __init__(self, num_features, hidden, num_head, alpha = 0.01, p = 0.5):
        super(GATLayer, self).__init__()
        self.num_features = num_features
        self.hidden = hidden
        self.num_head = num_head
        self.alpha = alpha

        self.gat = GATv2Conv(num_features, hidden, num_head, dropout = p, edge_dim = 2)
        self.norm  = nn.BatchNorm1d(num_head * hidden)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x : torch.Tensor, edge_idx : Optional[torch.Tensor]  = None, edge_attr : Optional[torch.Tensor] = None)->torch.Tensor:
        x = self.gat(x, edge_index = edge_idx, edge_attr = edge_attr)
        x = self.norm(x)
        x = self.act(x)
        return x

class GATNet(nn.Module):
    def __init__(self, output_dim : int = 2, hidden : int = 128, num_heads : int = 4, p : float = 0.25, alpha = 0.01, embedd_max_norm = 1.0, n_layers : int = 4):
        super(GATNet, self).__init__()
        torch.manual_seed(42)
        self.hidden = hidden
        self.alpha = alpha
        self.embedd_max_norm = embedd_max_norm
        self.output_dim = output_dim
   
        self.embedd = FeatureEmbedding(feature_lens = atom_feats, max_norm=embedd_max_norm)

        self.gc = nn.ModuleList()
        for i in range(n_layers):
            self.gc.append(GATLayer(sum(atom_feats) if i == 0 else hidden * num_heads, hidden, num_heads, alpha, p))

        self.mlp = nn.ModuleList(
            [
                nn.Linear(num_heads * self.hidden, self.hidden//2),
                nn.BatchNorm1d(self.hidden//2),
                nn.ReLU(),
                nn.Linear(self.hidden//2, self.hidden//2),
                nn.BatchNorm1d(self.hidden//2),
                nn.ReLU(),
                nn.Linear(self.hidden//2, output_dim),
            ]
        )
       
    def forward(self, inputs)->torch.Tensor:
        x, edge_idx, edge_attr, batch_idx = inputs.x,  inputs.edge_index, inputs.edge_attr, inputs.batch
        x = self.embedd(x)
        for layer in self.gc:
            x = layer(x, edge_idx, edge_attr)
        x = global_add_pool(x, batch_idx)

        for layer in self.mlp:
            x = layer(x)

        return x if self.output_dim != 1 else x.squeeze(1)


class ChebConvLayer(nn.Module):
    def __init__(self, n_dims_in : int, k : int, n_dims_out : int, alpha = 0.01):
        super(ChebConvLayer, self).__init__()
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out
        self.k = k
        self.alpha = alpha

        self.cheb = ChebConv(n_dims_in, n_dims_out, k, normalization='sym')
        self.norm = nn.BatchNorm1d(n_dims_out)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x, edge_idx = None, edge_attr = None):
        x = self.cheb(x,edge_idx, edge_attr)
        x = self.norm(x)
        x = self.act(x)
        return x

class ChebNet(nn.Module):
    def __init__(self, k : int, hidden : int, alpha = 0.01, embedd_max_norm = 1.0, n_layers : int = 4):
        super(ChebNet, self).__init__()
        self.k = k
        self.hidden = hidden
        self.alpha = alpha
        self.n_layers = n_layers

        self.embedd_max_norm = embedd_max_norm

        self.embedd = FeatureEmbedding(
            feature_lens=atom_feats, max_norm=embedd_max_norm)

        self.gc = nn.ModuleList()
        for i in range(n_layers):
            self.gc.append(ChebConvLayer(sum(atom_feats) if i == 0 else hidden, k, hidden, alpha))

        self.mlp = nn.ModuleList(
            [
                nn.Linear(self.hidden, self.hidden//2),
                nn.BatchNorm1d(self.hidden//2),
                nn.ReLU(),
                nn.Linear(self.hidden//2, self.hidden//2),
                nn.BatchNorm1d(self.hidden//2),
                nn.ReLU(),
                nn.Linear(self.hidden//2, 1),
            ]
        )

    def forward(self, inputs)->torch.Tensor:
        x, edge_idx, batch_idx = inputs.x,  inputs.edge_index, inputs.batch
        x = self.embedd(x)

        for layer in self.gc:
            x = layer(x, edge_idx)
        x = global_add_pool(x, batch_idx)

        for layer in self.mlp:
            x = layer(x)

        return x.squeeze(1)

    def summary(self, sample_inputs):
        print(summary(self, sample_inputs, max_depth = None, show_parent_layers=True, show_input = True))


class GINLayer(nn.Module):
    def __init__(self, num_features, hidden, eps : float = 0, train_eps : bool = True, alpha = 0.01):
        super(GINLayer, self).__init__()
        self.num_features = num_features
        self.hidden = hidden
        self.alpha = alpha

        self.nn = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(alpha),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(alpha),
        )

        self.conv = GINEConv(
            self.nn,
            eps = eps,
            train_eps = train_eps,
            edge_dim = 2
        )

        self.norm  = nn.BatchNorm1d(hidden)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x : torch.Tensor, edge_idx : Optional[torch.Tensor]  = None, edge_attr : Optional[torch.Tensor] = None)->torch.Tensor:
        x = self.conv(x, edge_index = edge_idx, edge_attr = edge_attr)
        x = self.norm(x)
        x = self.act(x)
        return x

class GINNet(nn.Module):
    def __init__(self, output_dim : int = 2, hidden : int = 128, eps : float = 0, train_eps : bool = True, alpha = 0.01, embedd_max_norm = 1.0, n_layers : int = 4):
        super(GINNet, self).__init__()
        torch.manual_seed(42)
        self.hidden = hidden
        self.alpha = alpha
        self.embedd_max_norm = embedd_max_norm
        self.output_dim = output_dim
   
        self.embedd = FeatureEmbedding(feature_lens = atom_feats, max_norm=embedd_max_norm)

        self.gc = nn.ModuleList()
        for i in range(n_layers):
            self.gc.append(GINLayer(sum(atom_feats) if i == 0 else hidden, hidden, eps, train_eps, alpha))

        self.mlp = nn.ModuleList(
            [
                nn.Linear(self.hidden, self.hidden//2),
                nn.BatchNorm1d(self.hidden//2),
                nn.ReLU(),
                nn.Linear(self.hidden//2, self.hidden//2),
                nn.BatchNorm1d(self.hidden//2),
                nn.ReLU(),
                nn.Linear(self.hidden//2, output_dim),
            ]
        )
       
    def forward(self, inputs)->torch.Tensor:
        x, edge_idx, edge_attr, batch_idx = inputs.x,  inputs.edge_index, inputs.edge_attr, inputs.batch
        x = self.embedd(x)
        for layer in self.gc:
            x = layer(x, edge_idx, edge_attr)
        x = global_add_pool(x, batch_idx)

        for layer in self.mlp:
            x = layer(x)

        return x if self.output_dim != 1 else x.squeeze(1)
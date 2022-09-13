import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Union, Tuple, Optional, List
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv, GCNConv, GATv2Conv, GINEConv, PNAConv
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.nn.pool.graclus import graclus
from torch_scatter import scatter_max
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch_geometric.nn import max_pool, max_pool_x, avg_pool, BatchNorm
from torch_geometric.utils.undirected import to_undirected
from pytorch_model_summary import summary
from src.preprocessing import ATOMS_LIST, ATOMS_DEGREE, ATOMS_NUMHS, ATOMS_VALENCE, ATOMS_AROMATIC, ATOMS_RING, ATOMS_HYBRID, BOND_TYPE, BOND_AROMATIC, BOND_CONJUGATED, BOND_RING, atom_properties

atom_feats = [
    len(ATOMS_LIST) + 1, 
    len(ATOMS_DEGREE) + 1, 
    len(ATOMS_NUMHS), 
    len(ATOMS_VALENCE), 
    len(ATOMS_AROMATIC),
    len(ATOMS_RING),
    len(ATOMS_HYBRID) + 1,
]

edge_feats = [
    len(list(BOND_TYPE.items())), 
    len(BOND_AROMATIC), 
    len(BOND_CONJUGATED),
    len(BOND_RING)
]

class FeatureEmbedding(nn.Module):
    def __init__(self, feature_lens : List, max_norm = 1.0):
        super(FeatureEmbedding, self).__init__()
        self.feature_lens = feature_lens
        self.emb_layers = nn.ModuleList()
        self.max_norm = max_norm

        for size in feature_lens:
            emb_layer = nn.Embedding(size, size, max_norm = max_norm)
            emb_layer.load_state_dict({'weight': torch.eye(size)})
            self.emb_layers.append(emb_layer)

    def forward(self, x : torch.Tensor):
        output = []
        for i, layer in enumerate(self.emb_layers):
            output.append(layer(x[:, i].long()))

        # Concatenate all node feature as dim 1
        output = torch.cat(output, 1)

        # normalization
        output = F.normalize(output)

        return output

class GraphEmbedding(nn.Module):
    def __init__(self, atom_feats : List, edge_feats : List, max_norm = 1.0):
        super(GraphEmbedding, self).__init__()
        self.emb_layers = nn.ModuleList()
        self.max_norm = max_norm
        self.atom_feats = atom_feats
        self.edge_feats = edge_feats

        for size in atom_feats + edge_feats:
            emb_layer = nn.Embedding(size, size, max_norm = max_norm)
            emb_layer.load_state_dict({'weight': torch.eye(size)})
            self.emb_layers.append(emb_layer)

    def forward(self, node : torch.Tensor, edge_attr : torch.Tensor):
        atom_feat = []
        edge_feat = []

        for i, layer in enumerate(self.emb_layers[0:len(self.atom_feats)]):
            atom_feat.append(layer(node[:, i].long()))

        for i, layer in enumerate(self.emb_layers[len(self.atom_feats):]):
            edge_feat.append(layer(edge_attr[:,i].long()))

        # Concatenate all feature as dim 1
        atom_feat = F.normalize(torch.cat(atom_feat, 1))
        edge_feat = F.normalize(torch.cat(edge_feat, 1))

        return atom_feat, edge_feat

class GCNLayer(nn.Module):
    def __init__(self, in_features : int, out_features : int, alpha : float)->None:
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.gconv = GCNConv(in_features, out_features)
        self.norm  = BatchNorm(out_features)
        self.act = nn.LeakyReLU(alpha)
        
    def forward(self, x : torch.Tensor, edge_idx : Optional[torch.Tensor]= None, edge_attr : Optional[torch.Tensor] = None)->torch.Tensor:
        h = self.gconv(x, edge_idx, edge_weight = edge_attr)
        h = self.norm(h)
        h = self.act(h)
        return h

class GConvNet(nn.Module):
    def __init__(self, hidden, output_dim : int = 2, alpha = 0.01, embedd_max_norm = 1.0, n_layers : int = 4, agg : Literal['pool','mean','add'] = 'add'):
        super(GConvNet, self).__init__()
        torch.manual_seed(42)
        self.hidden = hidden
        self.alpha = alpha
        self.embedd_max_norm = embedd_max_norm
        self.output_dim = output_dim
        self.agg = agg

        self.atom_embedd = FeatureEmbedding(feature_lens = atom_feats, max_norm=embedd_max_norm)
        self.edge_embedd = FeatureEmbedding(feature_lens = edge_feats, max_norm=embedd_max_norm)
        self.edge_weight = nn.Linear(sum(edge_feats),1)

        # self.embedd = GraphEmbedding(atom_feats, edge_feats, embedd_max_norm)

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
        x, edge_idx, edge_attr, batch_idx = inputs.x, inputs.edge_index, inputs.edge_attr, inputs.batch

        x = self.atom_embedd(x) 
        edge_attr = self.edge_embedd(edge_attr)
        edge_weight = self.edge_weight(edge_attr)

        for layer in self.gc:
            x = layer(x, edge_idx, edge_weight)

        if self.agg == 'add':
            x = global_add_pool(x, batch_idx)
        elif self.agg == 'mean':
            x = global_mean_pool(x, batch_idx)
        elif self.agg == 'pool':
            x = global_max_pool(x, batch_idx)

        for layer in self.mlp:
            x = layer(x)

        return x if self.output_dim != 1 else x.squeeze(1)

class GATLayer(nn.Module):
    def __init__(self, num_features, hidden, num_head, alpha = 0.01, p = 0.5, edge_dim : int = 2):
        super(GATLayer, self).__init__()
        self.num_features = num_features
        self.hidden = hidden
        self.num_head = num_head
        self.alpha = alpha

        self.gat = GATv2Conv(num_features, hidden, num_head, dropout = p, edge_dim = edge_dim)
        self.norm  = BatchNorm(num_head * hidden)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x : torch.Tensor, edge_idx : Optional[torch.Tensor]  = None, edge_attr : Optional[torch.Tensor] = None)->torch.Tensor:
        x = self.gat(x, edge_index = edge_idx, edge_attr = edge_attr)
        x = self.norm(x)
        x = self.act(x)
        return x

class GATNet(nn.Module):
    def __init__(self, output_dim : int = 2, hidden : int = 128, num_heads : int = 4, p : float = 0.25, alpha = 0.01, embedd_max_norm = 1.0, n_layers : int = 4, agg : Literal['pool', 'mean', 'add'] = 'add'):
        super(GATNet, self).__init__()
        torch.manual_seed(42)
        self.hidden = hidden
        self.alpha = alpha
        self.embedd_max_norm = embedd_max_norm
        self.output_dim = output_dim
        self.agg = agg
        self.edge_dim = sum(edge_feats)
   
        self.atom_embedd = FeatureEmbedding(feature_lens = atom_feats, max_norm=embedd_max_norm)
        self.edge_embedd = FeatureEmbedding(feature_lens = edge_feats, max_norm=embedd_max_norm)

        self.gc = nn.ModuleList()
        for i in range(n_layers):
            self.gc.append(GATLayer(sum(atom_feats) if i == 0 else hidden * num_heads, hidden, num_heads, alpha, p, sum(edge_feats)))

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

        x = self.atom_embedd(x)
        edge_attr = self.edge_embedd(edge_attr)

        for layer in self.gc:
            x = layer(x, edge_idx, edge_attr)

        if self.agg == 'add':
            x = global_add_pool(x, batch_idx)
        elif self.agg == 'mean':
            x = global_mean_pool(x, batch_idx)
        elif self.agg == 'pool':
            x = global_max_pool(x, batch_idx)
        
        for layer in self.mlp:
            x = layer(x)

        return x if self.output_dim != 1 else x.squeeze(1)

class GATNetHybrid(nn.Module):
    def __init__(self, output_dim : int = 2, hidden : int = 128, num_heads : int = 4, p : float = 0.25, alpha = 0.01, embedd_max_norm = 1.0, n_layers : int = 4, agg : Literal['pool', 'mean', 'add'] = 'add'):
        super(GATNetHybrid, self).__init__()
        torch.manual_seed(42)
        self.hidden = hidden
        self.alpha = alpha
        self.embedd_max_norm = embedd_max_norm
        self.output_dim = output_dim
        self.agg = agg
        self.edge_dim = sum(edge_feats)
   
        self.atom_embedd = FeatureEmbedding(feature_lens = atom_feats, max_norm=embedd_max_norm)
        self.edge_embedd = FeatureEmbedding(feature_lens = edge_feats, max_norm=embedd_max_norm)

        self.gc = nn.ModuleList()
        for i in range(n_layers):
            self.gc.append(GATLayer(sum(atom_feats) if i == 0 else hidden * num_heads, hidden, num_heads, alpha, p, sum(edge_feats) + 2))

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

        x_embed = self.atom_embedd(x[:,0:len(atom_feats)])
        edge_attr_embed = self.edge_embedd(edge_attr[:,0:len(edge_feats)])

        # x_linear = x[:,len(atom_feats):-1]
        edge_attr_linear = edge_attr[:,len(edge_feats):]

        x = x_embed
        # x = torch.cat([x_embed, x_linear], axis = 1)
        edge_attr = torch.cat([edge_attr_embed, edge_attr_linear], axis = 1)

        for layer in self.gc:
            x = layer(x, edge_idx, edge_attr)

        if self.agg == 'add':
            x = global_add_pool(x, batch_idx)
        elif self.agg == 'mean':
            x = global_mean_pool(x, batch_idx)
        elif self.agg == 'pool':
            x = global_max_pool(x, batch_idx)
        
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


class TransformerBasedLayer(nn.Module):
    def __init__(self, num_features : int , hidden : int , num_head : int, alpha = 0.01, p = 0.5):
        super(TransformerBasedLayer, self).__init__()
        self.num_features = num_features
        self.hidden = hidden
        self.num_head = num_head
        self.alpha = alpha

        self.trans = TransformerConv(num_features, hidden, num_head, True, False, p, edge_dim = 2)
        self.norm  = nn.BatchNorm1d(num_head * hidden)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x : torch.Tensor, edge_idx : Optional[torch.Tensor]  = None, edge_attr : Optional[torch.Tensor] = None)->torch.Tensor:
        x = self.trans(x, edge_index = edge_idx, edge_attr = edge_attr)
        x = self.norm(x)
        x = self.act(x)

        return x

class TransformerBasedModel(nn.Module):
    def __init__(self, output_dim : int = 2, hidden : int = 128, num_heads : int = 4, p : float = 0.25, alpha = 0.01, embedd_max_norm = 1.0, n_layers : int = 4):
        super(TransformerBasedModel, self).__init__()
        torch.manual_seed(42)
        self.hidden = hidden
        self.alpha = alpha
        self.embedd_max_norm = embedd_max_norm
        self.output_dim = output_dim
   
        self.embedd = FeatureEmbedding(feature_lens = atom_feats, max_norm=embedd_max_norm)

        self.gc = nn.ModuleList()
        for i in range(n_layers):
            self.gc.append(TransformerBasedLayer(sum(atom_feats) if i == 0 else hidden * num_heads, hidden, num_heads, alpha, p))

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

class PNALayer(nn.Module):
    def __init__(
        self, 
        num_features : int, 
        hidden : int, 
        aggregators : List, 
        scalers : List, 
        deg, 
        edge_dim : int = 2, 
        pre_layers : int = 1,
        post_layers : int = 1, 
        towers : int = 4,
        alpha : float = 0.1, 
        p : float = 0.25
        ):

        super(PNALayer, self).__init__()
        self.num_features = num_features
        self.hidden = hidden
        self.alpha = alpha
        self.p = p

        self.pna = PNAConv(num_features, hidden, aggregators, scalers, deg = deg, edge_dim = edge_dim, pre_layers=pre_layers, post_layers=post_layers, towers=towers)
        self.norm  = BatchNorm(hidden)
        self.act = nn.LeakyReLU(alpha)
        self.drop = nn.Dropout(p)

    def forward(self, x : torch.Tensor, edge_idx : Optional[torch.Tensor]  = None, edge_attr : Optional[torch.Tensor] = None)->torch.Tensor:
        x = self.pna(x, edge_index = edge_idx, edge_attr = edge_attr)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)

        return x

default_aggrs = ['sum', 'mean', 'min', 'max', 'std']
default_scalers = ['identity', 'amplification', 'attenuation']

class PNANet(nn.Module):
    def __init__(
        self, 
        deg, 
        aggrs = default_aggrs, 
        scalers = default_scalers, 
        output_dim : int = 2, 
        hidden : int = 128, 
        p : float = 0.25, 
        alpha = 0.01, 
        embedd_max_norm = 1.0, 
        n_layers : int = 4, 
        pre_layers : int = 1,
        post_layers : int = 1,
        towers : int = 4,
        ):
        
        super(PNANet, self).__init__()
        torch.manual_seed(42)
        self.hidden = hidden
        self.alpha = alpha
        self.embedd_max_norm = embedd_max_norm
        self.output_dim = output_dim
   
        self.atom_embedd = FeatureEmbedding(feature_lens = atom_feats, max_norm=embedd_max_norm)
        self.edge_embedd = FeatureEmbedding(feature_lens = edge_feats, max_norm = embedd_max_norm)

        self.gc = nn.ModuleList()
        for i in range(n_layers):
            self.gc.append(PNALayer(sum(atom_feats) if i == 0 else hidden, hidden, aggrs, scalers, deg, sum(edge_feats), pre_layers, post_layers, towers, alpha, p))

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
        x = self.atom_embedd(x)
        edge_attr = self.edge_embedd(edge_attr)

        for layer in self.gc:
            x = layer(x, edge_idx, edge_attr)

        x = global_add_pool(x, batch_idx)

        for layer in self.mlp:
            x = layer(x)

        return x if self.output_dim != 1 else x.squeeze(1)
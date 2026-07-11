import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_scatter import scatter_add

class EdgeModel(nn.Module):
    def __init__(self, node_dim=3, edge_dim=4, hidden_dim=64, groups=8):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Sequential(
            Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.GroupNorm(groups, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(groups, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

    def forward(self, src, dest, edge_attr):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self, node_dim=3, hidden_dim=64, groups=8):
        super(NodeModel, self).__init__()
        self.node_mlp = Sequential(
            Linear(node_dim + hidden_dim, hidden_dim),
            nn.GroupNorm(groups, hidden_dim),
            ReLU(),
            Linear(hidden_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg_messages = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, agg_messages], dim=1)
        return self.node_mlp(out)

class TrueTrackMLGNN(nn.Module):
    def __init__(self):
        super(TrueTrackMLGNN, self).__init__()
        self.edge_encoder = EdgeModel(node_dim=3, edge_dim=4, hidden_dim=64)
        self.node_updater = NodeModel(node_dim=3, hidden_dim=64)

        self.final_classifier = Sequential(
            Linear(64, 32),
            ReLU(),
            Linear(32, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        row, col = edge_index

        delta = x[row] - x[col]
        distance = torch.norm(delta, dim=1, keepdim=True)
        raw_edge_attr = torch.cat([delta, distance], dim=1)

        messages = self.edge_encoder(x[row], x[col], raw_edge_attr)
        updated_nodes = self.node_updater(x, edge_index, messages)
        final_edge_inputs = self.edge_encoder(updated_nodes[row], updated_nodes[col], raw_edge_attr)

        return self.final_classifier(final_edge_inputs).squeeze(-1)
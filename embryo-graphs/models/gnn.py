import torch
from torch.nn import Linear, ELU, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Sequential, GINConv
from torch_geometric.nn import global_max_pool, global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_convs):
        super(GNN, self).__init__()
        conv_modules = [
            (GINConv(torch.nn.Sequential(
                Linear(input_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ELU(inplace=True),
                Linear(hidden_channels, hidden_channels),
                ELU(inplace=True)
            )), 'x, edge_index -> x'),
        ]
        for i in range(num_convs):
            conv_modules.extend([
                (GINConv(torch.nn.Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ELU(inplace=True),
                Linear(hidden_channels, hidden_channels),
                ELU(inplace=True)
            )), 'x, edge_index -> x'),
            ])
        self.node_embedder = Sequential('x, edge_index', conv_modules)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.node_embedder(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
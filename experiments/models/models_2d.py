import torch
from torch.nn import GRU, Linear, ReLU, Sequential, LeakyReLU
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set, GATv2Conv

class Net(torch.nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr="mean")
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


class GAT(torch.nn.Module):

    def __init__(self, num_features, dim, heads=1, concat=True):
        super().__init__()
        self.conv1 = GATv2Conv(dim, dim, heads=heads, concat=concat)
        self.conv2 = GATv2Conv(dim * heads if concat else dim, dim*heads if concat else dim, heads=1, concat=False)
        self.conv3 = GATv2Conv(dim * heads if concat else dim, dim*heads, heads=1, concat=False)
        self.activation = LeakyReLU(negative_slope=0.2)
        self.in_linear = Linear(num_features, dim)
        
        self.mlp = Sequential(Linear(dim*heads if concat else dim, dim), ReLU(), Linear(dim, dim), ReLU(), Linear(dim, 1))
        
        # Add linear layers for skip connections
        self.skip1 = Linear(dim, dim * heads if concat else dim)

    def forward(self, data):
        x = F.relu(self.in_linear(data.x))
        
        # First GAT layer with skip connection
        out1 = F.relu(self.conv1(x, data.edge_index))
        out1 = out1 + self.skip1(x)
        
        # Second GAT layer with skip connection
        out2 = F.relu(self.conv2(out1, data.edge_index))
        out2 = out2 + out1  # Dimensions should match due to concat=False in conv2
        
        # Third GAT layer with skip connection
        out3 = F.relu(self.conv3(out2, data.edge_index))
        out3 = out3 + out2
        
        # Final MLP
        out = self.mlp(out3)
        return out.view(-1)
        
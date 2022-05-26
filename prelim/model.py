from torch.nn import Linear
import torch.nn.functional as F
from utils import *
from torch import nn
from torch.nn import Parameter
from source import GCNConv
from torch_geometric.nn import GINConv, SAGEConv
from torch.nn.utils import spectral_norm


class MLP_classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_classifier, self).__init__()
        self.args = args

        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = self.lin2(x)

        return x


class GCN_classifier(torch.nn.Module):
    def __init__(self, args):
        super(GCN_classifier, self).__init__()

        self.args = args

        self.lin1 = Linear(args.num_features, args.hidden, bias=False)

        self.lin2 = Linear(args.hidden, args.num_classes)

        self.bias = Parameter(torch.Tensor(args.num_classes))

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h = propagate(x, edge_index)
        h = self.lin1(h) + self.bias
        h = self.lin2(h)

        return h


class GIN_classifier(nn.Module):
    def __init__(self, args):
        super(GIN_classifier, self).__init__()

        self.args = args

        self.mlp = nn.Sequential(
            nn.Linear(args.num_features, args.hidden),
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Linear(args.hidden, args.hidden),
        )

        self.conv = GINConv(self.mlp)
        self.lin = Linear(args.hidden, args.num_classes)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.lin(h)
        return h

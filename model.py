from torch.nn import Linear
import torch.nn.functional as F
from utils import *
from torch import nn
from torch.nn import Parameter
from source import GCNConv
from torch_geometric.nn import GINConv, SAGEConv
from torch.nn.utils import spectral_norm


class channel_masker(nn.Module):
    def __init__(self, args):
        super(channel_masker, self).__init__()

        self.weights = nn.Parameter(torch.distributions.Uniform(
            0, 1).sample((args.num_features, 2)))

    def reset_parameters(self):
        self.weights = torch.nn.init.xavier_uniform_(self.weights)

    def forward(self):
        return self.weights


class MLP_discriminator(torch.nn.Module):
    def __init__(self, args):
        super(MLP_discriminator, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, 1)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None, mask_node=None):
        h = self.lin(h)

        return torch.sigmoid(h)


class MLP_encoder(torch.nn.Module):
    def __init__(self, args):
        super(MLP_encoder, self).__init__()
        self.args = args

        self.lin = Linear(args.num_features, args.hidden)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def forward(self, x, edge_index=None, mask_node=None):
        h = self.lin(x)

        return h


class GCN_encoder_scatter(torch.nn.Module):
    def __init__(self, args):
        super(GCN_encoder_scatter, self).__init__()

        self.args = args

        self.lin = Linear(args.num_features, args.hidden, bias=False)

        self.bias = Parameter(torch.Tensor(args.hidden))

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.lin(x)
        h = propagate2(h, edge_index) + self.bias

        return h


class GCN_encoder_spmm(torch.nn.Module):
    def __init__(self, args):
        super(GCN_encoder_spmm, self).__init__()

        self.args = args

        self.lin = Linear(args.num_features, args.hidden, bias=False)

        self.bias = Parameter(torch.Tensor(args.hidden))

    def clip_parameters(self, channel_weights):
        for i in range(self.lin.weight.data.shape[1]):
            self.lin.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                   self.args.clip_e * channel_weights[i])

        # self.lin.weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.lin.weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0.0)

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.lin(x)
        h = torch.spmm(adj_norm_sp, h) + self.bias
        # h = propagate2(h, edge_index) + self.bias

        return h


class GIN_encoder(nn.Module):
    def __init__(self, args):
        super(GIN_encoder, self).__init__()

        self.args = args

        self.mlp = nn.Sequential(
            nn.Linear(args.num_features, args.hidden),
            # nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            # nn.Linear(args.hidden, args.hidden),
        )

        self.conv = GINConv(self.mlp)

    def clip_parameters(self, channel_weights):
        for i in range(self.mlp[0].weight.data.shape[1]):
            self.mlp[0].weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                      self.args.clip_e * channel_weights[i])

        # self.mlp[0].weight.data[:,
        #                      channels].clamp_(-self.args.clip_e, self.args.clip_e)
        # self.mlp[0].weight.data.clamp_(-self.args.clip_e, self.args.clip_e)

        # for p in self.conv.parameters():
        #     p.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.conv(x, edge_index)
        return h


class SAGE_encoder(nn.Module):
    def __init__(self, args):
        super(SAGE_encoder, self).__init__()

        self.args = args

        self.conv1 = SAGEConv(args.num_features, args.hidden, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden),
            nn.Dropout(p=args.dropout)
        )
        self.conv2 = SAGEConv(args.hidden, args.hidden, normalize=True)
        self.conv2.aggr = 'mean'

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def clip_parameters(self, channel_weights):
        for i in range(self.conv1.lin_l.weight.data.shape[1]):
            self.conv1.lin_l.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                           self.args.clip_e * channel_weights[i])

        for i in range(self.conv1.lin_r.weight.data.shape[1]):
            self.conv1.lin_r.weight.data[:, i].data.clamp_(-self.args.clip_e * channel_weights[i],
                                                           self.args.clip_e * channel_weights[i])

        # for p in self.conv1.parameters():
        #     p.data.clamp_(-self.args.clip_e, self.args.clip_e)
        # for p in self.conv2.parameters():
        #     p.data.clamp_(-self.args.clip_e, self.args.clip_e)

    def forward(self, x, edge_index, adj_norm_sp):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        h = self.conv2(x, edge_index)
        return h


class MLP_classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_classifier, self).__init__()
        self.args = args

        self.lin = Linear(args.hidden, args.num_classes)

    def clip_parameters(self):
        for p in self.lin.parameters():
            p.data.clamp_(-self.args.clip_c, self.args.clip_c)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None):
        h = self.lin(h)

        return h

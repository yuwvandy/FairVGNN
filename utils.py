from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter
import random
import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import pandas as pd
from sklearn.manifold import TSNE


def propagate(x, edge_index, edge_weight=None):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    if(edge_weight == None):
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def propagate_mask(x, edge_index, mask_node=None):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(
        edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    if(mask_node == None):
        mask_node = torch.ones_like(x[:, 0])

    mask_node = mask_node[row]
    mask_node[row == col] = 1

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row] * \
        mask_node.view(-1, 1)

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def propagate2(x, edge_index):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(
        edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.allow_tf32 = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    # torch.use_deterministic_algorithms(True)


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) -
                 sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) -
                   sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()


def visual(model, data, sens, dataname):
    model.eval()

    print(data.y, sens)
    hidden = model.encoder(data.x, data.edge_index).cpu().detach().numpy()
    sens, data.y = sens.cpu().numpy(), data.y.cpu().numpy()
    idx_s0, idx_s1, idx_s2, idx_s3 = (sens == 0) & (data.y == 0), (sens == 0) & (
        data.y == 1), (sens == 1) & (data.y == 0), (sens == 1) & (data.y == 1)

    tsne_hidden = TSNE(n_components=2)
    tsne_hidden_x = tsne_hidden.fit_transform(hidden)

    tsne_input = TSNE(n_components=2)
    tsne_input_x = tsne_input.fit_transform(data.x.detach().cpu().numpy())

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    items = [tsne_input_x, tsne_hidden_x]
    names = ['input', 'hidden']

    for ax, item, name in zip(axs, items, names):
        ax.scatter(item[idx_s0][:, 0], item[idx_s0][:, 1], s=1,
                   c='red', marker='o', label='class 1, group1')
        ax.scatter(item[idx_s1][:, 0], item[idx_s1][:, 1], s=1,
                   c='blue', marker='o', label='class 2, group1')
        ax.scatter(item[idx_s2][:, 0], item[idx_s2][:, 1], s=10,
                   c='red', marker='', label='class 1, group2')
        ax.scatter(item[idx_s3][:, 0], item[idx_s3][:, 1], s=10,
                   c='blue', marker='+', label='class 2, group2')

        ax.set_title(name)
    ax.legend(frameon=0, loc='upper center',
              ncol=4, bbox_to_anchor=(-0.2, 1.2))

    plt.savefig(dataname + 'visual_tsne.pdf',
                dpi=1000, bbox_inches='tight')


def visual_sub(model, data, sens, dataname, k=50):
    idx_c1, idx_c2 = torch.where((sens == 0) == True)[
        0], torch.where((sens == 1) == True)[0]

    idx_subc1, idx_subc2 = idx_c1[torch.randperm(
        idx_c1.shape[0])[:k]], idx_c2[torch.randperm(idx_c2.shape[0])[:k]]

    idx_sub = torch.cat([idx_subc1, idx_subc2]).cpu().numpy()
    sens = sens[idx_sub]
    y = data.y[idx_sub]

    model.eval()

    hidden = model.encoder(data.x, data.edge_index).cpu().detach().numpy()
    sens, y = sens.cpu().numpy(), y.cpu().numpy()
    idx_s0, idx_s1, idx_s2, idx_s3 = (sens == 0) & (y == 0), (sens == 0) & (
        y == 1), (sens == 1) & (y == 0), (sens == 1) & (y == 1)

    tsne_hidden = TSNE(n_components=2)
    tsne_hidden_x = tsne_hidden.fit_transform(hidden)

    tsne_input = TSNE(n_components=2)
    tsne_input_x = tsne_input.fit_transform(data.x.detach().cpu().numpy())

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    items = [tsne_input_x[idx_sub], tsne_hidden_x[idx_sub]]
    names = ['input', 'hidden']

    for ax, item, name in zip(axs, items, names):
        ax.scatter(item[idx_s0][:, 0], item[idx_s0][:, 1], s=1,
                   c='red', marker='.', label='group1 class1')
        ax.scatter(item[idx_s1][:, 0], item[idx_s1][:, 1], s=5,
                   c='red', marker='*', label='group1 class2')
        ax.scatter(item[idx_s2][:, 0], item[idx_s2][:, 1], s=1,
                   c='blue', marker='.', label='group2 class1')
        ax.scatter(item[idx_s3][:, 0], item[idx_s3][:, 1], s=5,
                   c='blue', marker='*', label='group2 class2')

        ax.set_title(name)
    ax.legend(frameon=0, loc='upper center',
              ncol=4, bbox_to_anchor=(-0.2, 1.2))

    plt.savefig(dataname + 'visual_tsne.pdf',
                dpi=1000, bbox_inches='tight')


def pos_neg_mask(label, nodenum, train_mask):
    pos_mask = torch.stack([(label == label[i]).float()
                            for i in range(nodenum)])
    neg_mask = 1 - pos_mask

    return pos_mask[train_mask, :][:, train_mask], neg_mask[train_mask, :][:, train_mask]


def pos_neg_mask_sens(sens_label, label, nodenum, train_mask):
    pos_mask = torch.stack([((label == label[i]) & (sens_label == sens_label[i])).float()
                            for i in range(nodenum)])
    neg_mask = torch.stack([((label == label[i]) & (sens_label != sens_label[i])).float()
                            for i in range(nodenum)])

    return pos_mask[train_mask, :][:, train_mask], neg_mask[train_mask, :][:, train_mask]


def similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


def InfoNCE(h1, h2, pos_mask, neg_mask, tau=0.2):
    num_nodes = h1.shape[0]

    sim = similarity(h1, h2) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)

    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)

    return loss.mean()


def random_aug(x, edge_index, args):
    x_flip = flip_sens_feature(x, args.sens_idx, args.flip_node_ratio)

    edge_index1 = random_mask_edge(edge_index, args)
    edge_index2 = random_mask_edge(edge_index, args)

    mask1 = random_mask_node(x, args)
    mask2 = random_mask_node(x, args)

    return x_flip, edge_index1, edge_index2, mask1, mask2


def random_aug2(x, edge_index, args):
    # x_flip = flip_sens_feature(x, args.sens_idx, args.flip_node_ratio)
    edge_index = random_mask_edge(edge_index, args)

    mask = random_mask_node(x, args)

    return edge_index, mask


def flip_sens_feature(x, sens_idx, flip_node_ratio):
    node_num = x.shape[0]
    idx = np.arange(0, node_num)
    samp_idx = np.random.choice(idx, size=int(
        node_num * flip_node_ratio), replace=False)

    x_flip = x.clone()
    x_flip[:, sens_idx] = 1 - x_flip[:, sens_idx]

    return x_flip


def random_mask_edge(edge_index, args):
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        node_num = edge_index.size(0)
        edge_index = torch.stack([row, col], dim=0)

        edge_num = edge_index.shape[1]
        idx = np.arange(0, edge_num)
        samp_idx = np.random.choice(idx, size=int(
            edge_num * args.mask_edge_ratio), replace=False)

        mask = torch.ones(edge_num, dtype=torch.bool)
        mask[samp_idx] = 0

        edge_index = edge_index[:, mask]

        edge_index = SparseTensor(
            row=edge_index[0], col=edge_index[1],
            value=None, sparse_sizes=(node_num, node_num),
            is_sorted=True)

    else:
        edge_index, _ = add_remaining_self_loops(
            edge_index)
        edge_num = edge_index.shape[1]
        idx = np.arange(0, edge_num)
        samp_idx = np.random.choice(idx, size=int(
            edge_num * args.mask_edge_ratio), replace=False)

        mask = torch.ones_like(edge_index[0, :], dtype=torch.bool)
        mask[samp_idx] = 0

        edge_index = edge_index[:, mask]

    return edge_index


def random_mask_node(x, args):
    node_num = x.shape[0]
    idx = np.arange(0, node_num)
    samp_idx = np.random.choice(idx, size=int(
        node_num * args.mask_node_ratio), replace=False)

    mask = torch.ones_like(x[:, 0])
    mask[samp_idx] = 0

    return mask


def consis_loss(ps, temp=0.5):
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p

    avg_p = sum_p / len(ps)

    sharp_p = (torch.pow(avg_p, 1. / temp) /
               torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()

    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return 1 * loss


def sens_correlation(features, sens_idx):
    corr = pd.DataFrame(np.array(features)).corr()
    return corr[sens_idx].to_numpy()


def visualize(embeddings, y, s):
    X_embed = TSNE(n_components=2, learning_rate='auto',
                   init='random').fit_transform(embeddings)

    group1 = (y == 0) & (s == 0)
    group2 = (y == 0) & (s == 1)
    group3 = (y == 1) & (s == 0)
    group4 = (y == 1) & (s == 1)

    plt.scatter(X_embed[group1, 0], X_embed[group1, 1],
                s=5, c='tab:blue', marker='o')
    plt.scatter(X_embed[group2, 0], X_embed[group2, 1],
                s=5, c='tab:orange', marker='s')
    plt.scatter(X_embed[group3, 0], X_embed[group3, 1],
                s=5, c='tab:blue', marker='o')
    plt.scatter(X_embed[group4, 0], X_embed[group4, 1],
                s=5, c='tab:orange', marker='s')

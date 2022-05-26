from torch_geometric.utils import add_remaining_self_loops, degree, to_dense_adj
from torch_scatter import scatter
import random
import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import pandas as pd


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


def seed_everything(seed=0):
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.allow_tf32 = False

    # os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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


def covariance(features, sens_idx):
    features = features - features.mean(axis=0)

    return np.matmul(features.transpose(), features[:, sens_idx])


def correlation_var_visual(dataname, x, edge_index, k, sens_idx, header, top_k, select_k=None):
    prop_x = x
    for i in range(k):
        prop_x = propagate(prop_x, edge_index)

    prop_x2 = x
    for i in range(k + 1):
        prop_x2 = propagate(prop_x2, edge_index)

    corr_matrix = sens_correlation(np.array(x), sens_idx)
    corr_matrix_prop = sens_correlation(np.array(prop_x), sens_idx)
    corr_matrix_prop2 = sens_correlation(np.array(prop_x2), sens_idx)

    corr_idx = torch.tensor(np.argsort(-np.abs(corr_matrix)))
    corr_idx_prop = torch.tensor(np.argsort(-np.abs(corr_matrix_prop)))
    corr_idx_prop2 = torch.tensor(np.argsort(-np.abs(corr_matrix_prop2)))

    idx_mismatch = (corr_idx != corr_idx_prop)
    idx_mismatch2 = (corr_idx != corr_idx_prop2)
    color = np.array(['black' for i in range(len(idx_mismatch))])
    color[idx_mismatch] = 'red'

    X_axis = np.arange(0, 4 * len(header), 4)
    header = np.arange(0, len(header))
    plt.figure(figsize=(10, 5))

    bars = plt.bar(X_axis - 1, np.abs(corr_matrix), 1,
                   label='Original Features (S$_1$)')
    bars_prop = plt.bar(X_axis, np.abs(corr_matrix_prop), 1,
                        label='1-layer propagated Features (S$_2$)')
    bars_prop = plt.bar(X_axis + 1, np.abs(corr_matrix_prop2), 1,
                        label='2-layer propagated Features (S$_2$)')

    for index, val in enumerate(corr_idx.tolist()):
        plt.text(x=X_axis[val] - 1.7, y=np.abs(corr_matrix[val]) + 0.01,
                 s=f"{index}", fontdict=dict(fontsize=5), fontweight='bold', c='black')

    for index, val in enumerate(corr_idx_prop.tolist()):
        plt.text(x=X_axis[val] - 0.5, y=np.abs(corr_matrix_prop[val]) + 0.01,
                 s=f"{index}", fontdict=dict(fontsize=5), fontweight='bold', c='black')

    for index, val in enumerate(corr_idx_prop2.tolist()):
        plt.text(x=X_axis[val] + 0.8, y=np.abs(corr_matrix_prop2[val]) + 0.01,
                 s=f"{index}", fontdict=dict(fontsize=5), fontweight='bold', c='black')

    plt.xticks(X_axis, header, fontsize=15)
    plt.xlabel("Feature Dimension", fontsize=15, fontweight='bold')
    plt.ylabel("Linear Correlation to Sensitive Feature",
               fontsize=15, fontweight='bold')
    plt.legend(frameon=0, fontsize=15)

    plt.savefig(dataname + 'correlation_change' + str(k) + '.pdf',
                dpi=1000, bbox_inches='tight')


def correlation_var_visual_simple(dataname, x, edge_index, k, sens_idx, header, top_k, select_k):
    prop_xs = [x]
    for i in range(k):
        prop_xs.append(propagate(prop_xs[-1], edge_index))

    corr_matrices, corr_idxs = [], []
    for i in range(len(prop_xs)):
        prop_xs[i] = prop_xs[i][:, :select_k]
        corr_matrices.append(sens_correlation(np.array(prop_xs[i]), sens_idx))
        corr_idxs.append(torch.tensor(np.argsort(-np.abs(corr_matrices[i]))))

    idx_mismatch = (corr_idxs[0] != corr_idxs[1])
    for i in range(1, len(corr_idxs)):
        idx_mismatch = idx_mismatch | (corr_idxs[0] != corr_idxs[i])
    color = np.array(['black' for i in range(len(idx_mismatch))])
    color[idx_mismatch] = 'red'

    X_axis = np.arange(0, 8 * select_k, 8)
    header = np.arange(0, select_k)
    plt.figure(figsize=(4, 4))

    print(len(X_axis), len(corr_matrices[i]))

    bars = []
    for i in range(len(corr_matrices)):
        if(len(corr_matrices) % 2 == 0):
            start = -len(corr_matrices) // 2 + i + 0.5
        else:
            start = -len(corr_matrices) // 2 - 0.5 + i + 0.5

        bars.append(plt.bar(
            X_axis + start, np.abs(corr_matrices[i]), 1, label=str(i) + '-layer'))

    # for i in range(len(corr_matrices)):
    #     for index, val in enumerate(corr_idxs[i].tolist()):
    #         plt.text(x=X_axis[val] - 1.2, y=np.abs(corr_matrices[i][val]) + 0.01,
    #                  s=f"{index}", fontdict=dict(fontsize=10), fontweight='bold', c=color[index])

    plt.xticks(X_axis - 1, header, fontsize=10)
    plt.xlabel("Feature channel", fontsize=15, fontweight='bold')
    plt.ylabel(r"Sensitive correlation $|\rho_i|$",
               fontsize=15, fontweight='bold')
    plt.legend(frameon=0, fontsize=15)

    plt.savefig(dataname + 'correlation_change_simple.pdf',
                dpi=1000, bbox_inches='tight')


def channel_homophily(dataname, x, edge_index, sens_idx, prop_k):
    x = (x - torch.mean(x, dim=0)) / \
        torch.sum((x - torch.mean(x, dim=0))**2, dim=0)**0.5
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
    adj_matrix = to_dense_adj(edge_index)[0]

    degree = torch.diag(1 / adj_matrix.sum(dim=0))
    norm_adj_matrix = torch.matmul(torch.matmul(
        degree**0.5, adj_matrix), degree**0.5)
    norm_adj_matrix2 = torch.matmul(norm_adj_matrix, norm_adj_matrix)

    sens_homophily = []
    for i in range(x.shape[1]):
        sens_homophily.append(torch.matmul(torch.matmul(
            x[:, sens_idx], norm_adj_matrix2), x[:, i]).item())

    print(sens_homophily)

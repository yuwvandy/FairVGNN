import pandas as pd
import os
import numpy as np
import random
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
import torch
from utils import sens_correlation
import scipy.sparse as sp


def index_to_mask(node_num, index):
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = 1

    return mask


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(
        1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map = np.array(idx_map)

    return idx_map


def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="dataset/credit/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    # sensitive feature removal
    # header.remove('Age')

#    # Normalize MaxBillAmountOverLast6Months
#    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
#
#    # Normalize MaxPaymentAmountOverLast6Months
#    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
#
#    # Normalize MostRecentBillAmount
#    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
#
#    # Normalize MostRecentPaymentAmount
#    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
#
#    # Normalize TotalMonthsOverdue
#    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    # print(features)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    #
    #

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(
        label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(
        0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens


def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="dataset/bail/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    # sensitive feature removal
    # header.remove('WHITE')

    # # Normalize School
    # idx_features_labels['SCHOOL'] = 2*(idx_features_labels['SCHOOL']-idx_features_labels['SCHOOL'].min()).div(idx_features_labels['SCHOOL'].max() - idx_features_labels['SCHOOL'].min()) - 1

    # # Normalize RULE
    # idx_features_labels['RULE'] = 2*(idx_features_labels['RULE']-idx_features_labels['RULE'].min()).div(idx_features_labels['RULE'].max() - idx_features_labels['RULE'].min()) - 1

    # # Normalize AGE
    # idx_features_labels['AGE'] = 2*(idx_features_labels['AGE']-idx_features_labels['AGE'].min()).div(idx_features_labels['AGE'].max() - idx_features_labels['AGE'].min()) - 1

    # # Normalize TSERVD
    # idx_features_labels['TSERVD'] = 2*(idx_features_labels['TSERVD']-idx_features_labels['TSERVD'].min()).div(idx_features_labels['TSERVD'].max() - idx_features_labels['TSERVD'].min()) - 1

    # # Normalize FOLLOW
    # idx_features_labels['FOLLOW'] = 2*(idx_features_labels['FOLLOW']-idx_features_labels['FOLLOW'].min()).div(idx_features_labels['FOLLOW'].max() - idx_features_labels['FOLLOW'].min()) - 1

    # # Normalize TIME
    # idx_features_labels['TIME'] = 2*(idx_features_labels['TIME']-idx_features_labels['TIME'].min()).div(idx_features_labels['TIME'].max() - idx_features_labels['TIME'].min()) - 1

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    # print(features)

    # features = normalize(features)
    # adj = adj + sp.eye(adj.shape[0])

    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(
        label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(
        0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens


def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="dataset/german/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(
        os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender']
                                  == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

#    for i in range(idx_features_labels['PurposeOfLoan'].unique().shape[0]):
#        val = idx_features_labels['PurposeOfLoan'].unique()[i]
#        idx_features_labels['PurposeOfLoan'][idx_features_labels['PurposeOfLoan'] == val] = i

#    # Normalize LoanAmount
#    idx_features_labels['LoanAmount'] = 2*(idx_features_labels['LoanAmount']-idx_features_labels['LoanAmount'].min()).div(idx_features_labels['LoanAmount'].max() - idx_features_labels['LoanAmount'].min()) - 1
#
#    # Normalize Age
#    idx_features_labels['Age'] = 2*(idx_features_labels['Age']-idx_features_labels['Age'].min()).div(idx_features_labels['Age'].max() - idx_features_labels['Age'].min()) - 1
#
#    # Normalize LoanDuration
#    idx_features_labels['LoanDuration'] = 2*(idx_features_labels['LoanDuration']-idx_features_labels['LoanDuration'].min()).div(idx_features_labels['LoanDuration'].max() - idx_features_labels['LoanDuration'].min()) - 1
#
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(
            f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(
            idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)

    edge_index, _ = from_scipy_sparse_matrix(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)
    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(
        label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(
        0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.LongTensor(sens)
    train_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_train))
    val_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_val))
    test_mask = index_to_mask(features.shape[0], torch.LongTensor(idx_test))

    return adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens


def get_dataset(dataname, top_k):
    if(dataname == 'credit'):
        load, label_num = load_credit, 6000
    elif(dataname == 'bail'):
        load, label_num = load_bail, 100
    elif(dataname == 'german'):
        load, label_num = load_german, 100

    adj_norm_sp, edge_index, features, labels, train_mask, val_mask, test_mask, sens = load(
        dataset=dataname, label_number=label_num)

    if(dataname == 'credit'):
        sens_idx = 1
    elif(dataname == 'bail' or dataname == 'german'):
        sens_idx = 0

    x_max, x_min = torch.max(features, dim=0)[
        0], torch.min(features, dim=0)[0]

    if(dataname != 'german'):
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    corr_matrix = sens_correlation(features, sens_idx)
    corr_idx = np.argsort(-np.abs(corr_matrix))
    if(top_k > 0):
        # corr_idx = np.concatenate((corr_idx[:top_k], corr_idx[-top_k:]))
        corr_idx = corr_idx[:top_k]

    return Data(x=features, edge_index=edge_index, adj_norm_sp=adj_norm_sp, y=labels.float(), train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, sens=sens), sens_idx, corr_matrix, corr_idx, x_min, x_max

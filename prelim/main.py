from dataset import *
from model import *
from utils import *
from learn import *

import argparse
from tqdm import tqdm
from torch import tensor
import warnings
warnings.filterwarnings('ignore')
import math


def run(data, args):
    pbar = tqdm(range(args.runs), unit='run')

    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    sens_y = np.array(data.x[:, args.sens_idx])

    if(args.order == 's1'):
        x = data.x

    elif(args.order == 's2'):
        x = data.x
        for i in range(args.prop_k):
            x = propagate(x, data.edge_index)

    corr_matrix = sens_correlation(np.array(x), args.sens_idx)

    if args.type == 'single':
        if args.top_k != 0:
            corr_idx = torch.tensor(
                np.argsort(-np.abs(corr_matrix))[args.top_k - 1])
            print('mask channel:', corr_idx.tolist())
            data.x[:, corr_idx] = 0
    else:
        if args.top_k != 0:
            corr_idx = torch.tensor(
                np.argsort(-np.abs(corr_matrix))[:args.top_k])
            print('mask channel:', corr_idx.tolist())
            data.x[:, corr_idx] = 0

    data = data.to(args.device)
    x = x.to(args.device)

    if args.use_feature == 'after_prop':
        input_feat = x
    else:
        input_feat = data.x

    if(args.model == 'GCN'):
        model = GCN_classifier(args).to(args.device)
        optimizer = torch.optim.Adam([
            dict(params=model.lin1.parameters(), weight_decay=args.wd1),
            dict(params=model.lin2.parameters(), weight_decay=args.wd2),
            dict(params=model.bias, weight_decay=args.wd2)], lr=args.lr)
    elif(args.model == 'MLP'):
        model = MLP_classifier(args).to(args.device)
        optimizer = torch.optim.Adam([
            dict(params=model.lin1.parameters(), weight_decay=args.wd1),
            dict(params=model.lin2.parameters(), weight_decay=args.wd2)], lr=args.lr)
    elif(args.model == 'GIN'):
        model = GIN_classifier(args).to(args.device)
        optimizer = torch.optim.Adam([
            dict(params=model.lin.parameters(), weight_decay=args.wd1),
            dict(params=model.conv.parameters(), weight_decay=args.wd2)], lr=args.lr)

    for count in pbar:
        seed_everything(count + args.seed)

        model.reset_parameters()

        best_val_loss, best_val_f1, best_val_roc_auc = math.inf, 0, 0
        val_loss_history, val_f1_history, val_roc_auc_history = [], [], []

        for epoch in range(0, args.epochs):
            model.train()
            optimizer.zero_grad()

            output = model(input_feat, data.edge_index)

            loss_ce = F.binary_cross_entropy_with_logits(
                output[data.train_mask], data.y[data.train_mask].unsqueeze(1))

            loss_ce.backward()
            optimizer.step()

            accs, auc_rocs, F1s, parities, equalities, loss_val, embedding_tmp = evaluate_exploration(
                input_feat, model, data, args)

            if loss_val < best_val_loss:
                best_val_f1 = F1s['val']
                best_val_roc_auc = auc_rocs['val']
                best_val_loss = loss_val

                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']

                test_parity, test_equality = parities['test'], equalities['test']
                embedding = embedding_tmp

        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality

        # print(np.mean(auc_roc[:(count + 1)]), np.mean(f1[:(count + 1)]), np.mean(
        #     acc[:(count + 1)]), np.mean(parity[:(count + 1)]), np.mean(equality[:(count + 1)]))

    return acc, f1, auc_roc, parity, equality


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd1', type=float, default=1e-5)
    parser.add_argument('--wd2', type=float, default=1e-5)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--order', type=str, default='s1')
    parser.add_argument('--top_k', type=int, default=4)
    parser.add_argument('--prop_k', type=int, default=1)
    parser.add_argument('--type', type=str, default='single')
    parser.add_argument('--use_feature', type=str, default='before_prop')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # import the dataset
    data, args.sens_idx, args.header = get_dataset(args.dataset, args.top_k)
    args.num_features, args.num_classes = data.x.shape[1], len(
        data.y.unique()) - 1

    # cor_sens = covariance(data.x, args.sens_idx)
    # print(cor_sens)
    # correlation_var_visual(args.dataset, data.x,
    #                        data.edge_index, 4, args.sens_idx, args.header, args.top_k, -1)

    # correlation_var_visual_simple(args.dataset, data.x, data.edge_index,
    #                               4, args.sens_idx, args.header, args.top_k, 10)

    # channel_homophily(args.dataset, data.x, data.edge_index,
    #                   args.sens_idx, args.prop_k)

    # print((data.y == 1).sum(), (data.y == 0).sum())
    # print((data.y[data.train_mask] == 1).sum(),
    #       (data.y[data.train_mask] == 0).sum())
    # print((data.y[data.val_mask] == 1).sum(),
    #       (data.y[data.val_mask] == 0).sum())
    # print((data.y[data.test_mask] == 1).sum(),
    #       (data.y[data.test_mask] == 0).sum())

    # corr_matrix = sens_correlation(np.array(data.x), args.sens_idx)
    # corr_idx = torch.tensor(np.argsort(-np.abs(corr_matrix))[:args.top_k])
    # print(corr_idx)
    # print(corr_matrix[corr_idx[-1]])
    acc, f1, auc_roc, parity, equality = run(
        data, args)

    print('======' + args.dataset + '======')
    print(args.model)
    print('auc_roc: {:.2f}, {:.2f}'.format(
        np.mean(auc_roc) * 100, np.std(auc_roc) * 100))
    print('f1: {:.2f}, {:.2f}'.format(np.mean(f1) * 100, np.std(f1) * 100))
    print('Acc: {:.2f}, {:.2f}'.format(np.mean(acc) * 100, np.std(acc) * 100))
    print('parity: {:.2f}, {:.2f}'.format(
        np.mean(parity) * 100, np.std(parity) * 100))
    print('equality: {:.2f}, {:.2f}'.format(
        np.mean(equality) * 100, np.std(equality) * 100))

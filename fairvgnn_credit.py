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

from torch.optim.lr_scheduler import ExponentialLR


def run(data, args):
    pbar = tqdm(range(args.runs), unit='run')
    criterion = nn.BCELoss()
    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    data = data.to(args.device)

    generator = channel_masker(args).to(args.device)
    optimizer_g = torch.optim.Adam([
        dict(params=generator.weights, weight_decay=args.g_wd)], lr=args.g_lr)

    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = torch.optim.Adam([
        dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)], lr=args.d_lr)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = torch.optim.Adam([
        dict(params=classifier.lin.parameters(), weight_decay=args.c_wd)], lr=args.c_lr)

    if(args.encoder == 'MLP'):
        encoder = MLP_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'GCN'):
        if args.prop == 'scatter':
            encoder = GCN_encoder_scatter(args).to(args.device)
        else:
            encoder = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.bias, weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'GIN'):
        encoder = GIN_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'SAGE'):
        encoder = SAGE_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv1.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.conv2.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)

    for count in pbar:
        seed_everything(count + args.seed)
        generator.reset_parameters()
        discriminator.reset_parameters()
        classifier.reset_parameters()
        encoder.reset_parameters()

        best_val_tradeoff = 0
        best_val_loss = math.inf
        for epoch in range(0, args.epochs):
            if(args.f_mask == 'yes'):
                generator.eval()
                feature_weights, masks, = generator(), []
                for k in range(args.K):
                    mask = F.gumbel_softmax(
                        feature_weights, tau=1, hard=False)[:, 0]
                    masks.append(mask)

            # train discriminator to recognize the sensitive group
            discriminator.train()
            encoder.eval()
            for epoch_d in range(0, args.d_epochs):
                optimizer_d.zero_grad()

                if(args.f_mask == 'yes'):
                    loss_d = 0

                    for k in range(args.K):
                        x = data.x * masks[k].detach()
                        h = encoder(x, data.edge_index, data.adj_norm_sp)
                        output = discriminator(h)

                        loss_d += criterion(output.view(-1),
                                            data.x[:, args.sens_idx])

                    loss_d = loss_d / args.K
                else:
                    h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    output = discriminator(h)

                    loss_d = criterion(output.view(-1),
                                       data.x[:, args.sens_idx])

                loss_d.backward()
                optimizer_d.step()

            # train classifier
            classifier.train()
            encoder.train()
            for epoch_c in range(0, args.c_epochs):
                optimizer_c.zero_grad()
                optimizer_e.zero_grad()

                if(args.f_mask == 'yes'):
                    loss_c = 0
                    for k in range(args.K):
                        x = data.x * masks[k].detach()
                        h = encoder(x, data.edge_index, data.adj_norm_sp)
                        output = classifier(h)

                        loss_c += F.binary_cross_entropy_with_logits(
                            output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))

                    loss_c = loss_c / args.K

                else:
                    h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    output = classifier(h)

                    loss_c = F.binary_cross_entropy_with_logits(
                        output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))

                loss_c.backward()

                optimizer_e.step()
                optimizer_c.step()

            # train generator to fool discriminator
            generator.train()
            encoder.train()
            discriminator.eval()
            for epoch_g in range(0, args.g_epochs):
                optimizer_g.zero_grad()
                optimizer_e.zero_grad()

                if(args.f_mask == 'yes'):
                    loss_g = 0
                    feature_weights = generator()
                    for k in range(args.K):
                        mask = F.gumbel_softmax(
                            feature_weights, tau=1, hard=False)[:, 0]

                        x = data.x * mask
                        h = encoder(x, data.edge_index, data.adj_norm_sp)
                        output = discriminator(h)

                        loss_g += F.mse_loss(output.view(-1),
                                             0.5 * torch.ones_like(output.view(-1))) + args.ratio * F.mse_loss(mask.view(-1), torch.ones_like(mask.view(-1)))

                    loss_g = loss_g / args.K
                else:
                    h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    output = discriminator(h)

                    loss_g = F.mse_loss(output.view(-1),
                                        0.5 * torch.ones_like(output.view(-1)))

                loss_g.backward()

                optimizer_g.step()
                optimizer_e.step()

            if(args.weight_clip == 'yes'):
                if(args.f_mask == 'yes'):
                    weights = torch.stack(masks).mean(dim=0)
                else:
                    weights = torch.ones_like(data.x[0])

                encoder.clip_parameters(weights)
                classifier.clip_parameters()

            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(
                data.x, classifier, discriminator, generator, encoder, data, args)

            # print(epoch, 'Acc:', accs['test'], 'AUC_ROC:', auc_rocs['test'], 'F1:', F1s['test'],
            #       'Parity:', tmp_parity['test'], 'Equality:', tmp_equality['test'])

            # print(classifier.lin.weight.sum())
            # print(encoder.lin.weight.sum())

            if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

                best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
                    accs['val'] - (tmp_parity['val'] + tmp_equality['val'])

                # print('=====VALIDATION=====', epoch, epoch_g)
                # print('Utility:', auc_rocs['val'] + F1s['val'] + accs['val'],
                #       'Fairness:', tmp_parity['val'] + tmp_equality['val'])

                # print('=====VALIDATION-BEST=====', epoch, epoch_g)
                # print('Utility:', args.best_val_model_utility,
                #       'Fairness:', args.best_val_fair)

                # print('=====TEST=====', epoch)
                # print('Acc:', test_acc, 'AUC_ROC:', test_auc_roc, 'F1:', test_f1,
                #       'Parity:', test_parity, 'Equality:', test_equality)

                # print('=====epoch:{}====='.format(epoch))
                # print('sens_acc:', (((output.view(-1) > 0.5) & (data.x[:, args.sens_idx] == 1)).sum() + ((output.view(-1) < 0.5) &
                #                                                                                          (data.x[:, args.sens_idx] == 0)).sum()).item() / len(data.y))

        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality

        # print('auc_roc:', np.mean(auc_roc[:(count + 1)]))
        # print('f1:', np.mean(f1[:(count + 1)]))
        # print('acc:', np.mean(acc[:(count + 1)]))
        # print('Statistical parity:', np.mean(parity[:(count + 1)]))
        # print('Equal Opportunity:', np.mean(equality[:(count + 1)]))

    return acc, f1, auc_roc, parity, equality


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--d_epochs', type=int, default=5)
    parser.add_argument('--g_epochs', type=int, default=5)
    parser.add_argument('--c_epochs', type=int, default=5)
    parser.add_argument('--g_lr', type=float, default=0.001)
    parser.add_argument('--g_wd', type=float, default=0)
    parser.add_argument('--d_lr', type=float, default=0.001)
    parser.add_argument('--d_wd', type=float, default=0)
    parser.add_argument('--c_lr', type=float, default=0.001)
    parser.add_argument('--c_wd', type=float, default=0)
    parser.add_argument('--e_lr', type=float, default=0.001)
    parser.add_argument('--e_wd', type=float, default=0)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--prop', type=str, default='scatter')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='GIN')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--clip_e', type=float, default=1)
    parser.add_argument('--clip_c', type=float, default=1)
    parser.add_argument('--f_mask', type=str, default='yes')
    parser.add_argument('--weight_clip', type=str, default='yes')
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, args.sens_idx, args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_dataset(
        args.dataset, args.top_k)
    args.num_features, args.num_classes = data.x.shape[1], len(
        data.y.unique()) - 1

    # print((data.y == 1).sum(), (data.y == 0).sum())
    # print((data.y[data.train_mask] == 1).sum(),
    #       (data.y[data.train_mask] == 0).sum())
    # print((data.y[data.val_mask] == 1).sum(),
    #       (data.y[data.val_mask] == 0).sum())
    # print((data.y[data.test_mask] == 1).sum(),
    #       (data.y[data.test_mask] == 0).sum())

    args.train_ratio, args.val_ratio = torch.tensor([
        (data.y[data.train_mask] == 0).sum(), (data.y[data.train_mask] == 1).sum()]), torch.tensor([
            (data.y[data.val_mask] == 0).sum(), (data.y[data.val_mask] == 1).sum()])
    args.train_ratio, args.val_ratio = torch.max(
        args.train_ratio) / args.train_ratio, torch.max(args.val_ratio) / args.val_ratio
    args.train_ratio, args.val_ratio = args.train_ratio[
        data.y[data.train_mask].long()], args.val_ratio[data.y[data.val_mask].long()]

    # print(args.val_ratio, data.y[data.val_mask])

    acc, f1, auc_roc, parity, equality = run(data, args)
    print('======' + args.dataset + args.encoder + '======')
    print('auc_roc:', np.mean(auc_roc) * 100, np.std(auc_roc) * 100)
    print('Acc:', np.mean(acc) * 100, np.std(acc) * 100)
    print('f1:', np.mean(f1) * 100, np.std(f1) * 100)
    print('parity:', np.mean(parity) * 100, np.std(parity) * 100)
    print('equality:', np.mean(equality) * 100, np.std(equality) * 100)

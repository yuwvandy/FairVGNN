import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score, roc_auc_score
from utils import fair_metric, InfoNCE, random_aug, consis_loss


def evaluate_exploration(x, model, data, args):
    model.eval()

    with torch.no_grad():
        output = model(x, data.edge_index)

        loss_val = F.binary_cross_entropy_with_logits(
            output[data.val_mask], data.y[data.val_mask].unsqueeze(1))

    accs, auc_rocs, F1s, parities, equalities = {}, {}, {}, {}, {}

    pred_val = (output[data.val_mask].squeeze() > 0).type_as(data.y)
    pred_test = (output[data.test_mask].squeeze() > 0).type_as(data.y)

    accs['val'] = pred_val.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum()
    accs['test'] = pred_test.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum()

    F1s['val'] = f1_score(data.y[data.val_mask].cpu(
    ).numpy(), pred_val.cpu().numpy())

    F1s['test'] = f1_score(data.y[data.test_mask].cpu(
    ).numpy(), pred_test.cpu().numpy())

    auc_rocs['val'] = roc_auc_score(
        data.y[data.val_mask].cpu().numpy(), output[data.val_mask].detach().cpu().numpy())
    auc_rocs['test'] = roc_auc_score(
        data.y[data.test_mask].cpu().numpy(), output[data.test_mask].detach().cpu().numpy())

    parities['val'], equalities['val'] = fair_metric(pred_val.cpu().numpy(), data.y[data.val_mask].cpu(
    ).numpy(), data.sens[data.val_mask].cpu().numpy())
    parities['test'], equalities['test'] = fair_metric(pred_test.cpu().numpy(), data.y[data.test_mask].cpu(
    ).numpy(), data.sens[data.test_mask].cpu().numpy())

    return accs, auc_rocs, F1s, parities, equalities, loss_val, output

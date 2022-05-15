import torch
import numpy as np
import torch.nn.functional as F


def get_ood_scores(args, net, loader, ood_num_examples, device, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.to(device)
            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.score == 'energy':
                all_score = -to_np(args.T * torch.logsumexp(output / args.T, dim=1))
            else:
                all_score = -np.max(to_np(F.softmax(output/args.T, dim=1)), axis=1)

            _score.append(all_score)
            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(all_score[right_indices])
                _wrong_score.append(all_score[wrong_indices])

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def get_ood_gradnorm(args, net, loader, ood_num_examples, device, in_dist=False, print_norm=False):
    _score = []
    _right_score = []
    _wrong_score = []

    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)
    for batch_idx, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            break
        data = data.to(device)
        net.zero_grad()
        output = net(data)
        num_classes = output.shape[-1]
        targets = torch.ones((data.shape[0], num_classes)).to(device)
        output = output / args.T
        loss = torch.mean(torch.sum(-targets * logsoftmax(output), dim=-1))

        loss.backward()
        layer_grad = net.fc.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        all_score = -layer_grad_norm
        _score.append(all_score)

    if in_dist:
        return np.array(_score).copy()
    else:
        return np.array(_score)[:ood_num_examples].copy()

def get_calibration_scores(args, net, loader, device):
    logits_list = []
    labels_list = []

    from common.loss_function import _ECELoss
    ece_criterion = _ECELoss(n_bins=15)
    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]

            data = data.to(device)
            label = target.to(device)
            logits = net(data)

            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
    ece_error = ece_criterion(logits, labels, args.T)
    return ece_error

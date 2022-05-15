import numpy as np
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from common.ood_tools import get_ood_gradnorm

from models.utils import build_model
from datasets.utils import build_dataset
from common.ood_tools import get_ood_scores
if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from common.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import common.score_calculation as lib
parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--method_name', '-s', type=str, default='cifar10_clean_00_allconv_standard', help='Method name.')

parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--score', default='MSP', type=str, help='score options: Odin|MSP|energy|gradnorm')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--gpu', type=str, default="0", help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--seed', type=int, default=1, help='0 = CPU.')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')

parser.add_argument('--include_train', '-t', action='store_true',
                    help='test model on train set')
parser.add_argument('--optimal_t', action='store_true',
                    help='test model on train set')

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = True  # fire on all cylinders

if args.gpu is not None:
    if len(args.gpu) == 1:
        device = torch.device('cuda:{}'.format(int(args.gpu)))
    else:
        device = torch.device('cuda:0')
    torch.cuda.manual_seed(args.seed)
else:
    device = torch.device('cpu')


test_data, num_classes = build_dataset(args.dataset, mode="test")

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
model_type = args.method_name.split("_", 5)[-3]
alg = args.method_name.split("_", 5)[-1]
net = build_model(model_type, num_classes, device, args)

start_epoch = 0
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(os.path.join(args.load, alg), args.method_name + '_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name, map_location=device))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"
start_epoch = 0

net.eval()
cudnn.benchmark = True  # fire on all cylinders


if args.optimal_t:
    from common.utils import get_optimal_temperature
    args.T = get_optimal_temperature(net, test_loader, device)


# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

if args.score == 'Odin':
    # separated because no grad is not applied
    in_score, right_score, wrong_score = lib.get_ood_scores_odin(test_loader, net, args.test_bs, ood_num_examples, args.T, args.noise, device, in_dist=True)
elif args.score == 'gradnorm':
    _, right_score, wrong_score = get_ood_scores(args, net, test_loader, ood_num_examples, device, in_dist=True)
    in_score = get_ood_gradnorm(args, net, test_loader, ood_num_examples, device, in_dist=True, print_norm=args.print_norm)
else:
    in_score, right_score, wrong_score = get_ood_scores(args, net, test_loader, ood_num_examples, device, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))


# /////////////// End Detection Prelims ///////////////

print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=args.method_name)





# /////////////// ECE Detection ///////////////

print('\n\nECE Error')
from common.ood_tools import get_calibration_scores

ece_error = get_calibration_scores(args, net, test_loader, device)
print('ECE Error {:.2f}'.format(100 * ece_error.item()))

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):

        if args.score == 'Odin':
            out_score = lib.get_ood_scores_odin(ood_loader, net, args.test_bs, ood_num_examples, args.T, args.noise, device)
        elif args.score == 'gradnorm':
            out_score = get_ood_gradnorm(args, net, ood_loader, ood_num_examples, device, print_norm=args.print_norm)
        else:
            out_score = get_ood_scores(args, net, ood_loader, ood_num_examples, device)

        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
            
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        print_measures(auroc, aupr, fpr, args.method_name)



if __name__ == '__main__':
    OOD_data_list = [ "Textures", "SVHN", "LSUN-C", "LSUN-R", "iSUN", "Places365"]

    for data_name in OOD_data_list:
        if data_name == args.dataset:
            continue
        ood_data, _ = build_dataset(data_name, mode="test")
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                                     num_workers=args.prefetch, pin_memory=True)
        print('\n\n{} Detection'.format(data_name))
        get_and_print_results(ood_loader)

    # /////////////// Mean Results ///////////////
    print('\n\nMean Test Results')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)







from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy import misc

to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

def get_ood_scores_odin(loader, net, bs, ood_num_examples, T, noise, device, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        if batch_idx >= ood_num_examples // bs and in_dist is False:
            break
        data = data.to(device)
        data = Variable(data, requires_grad = True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODIN(data, output,net, T, noise, device)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def ODIN(inputs, outputs, model, temper, noiseMagnitude1, device):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
    #gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0/255.0))
    #gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1/255.0))
    #gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7/255.0))

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs
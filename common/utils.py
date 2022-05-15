import numpy as np

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


import os
import os.path
import copy
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal
import torch
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from common.loss_function import _ECELoss

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def build_for_cifar100(size, noise):
    """ The noise matrix flips to the "next" class with probability 'noise'.
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P



def train_valid_split(data, split_fold=4):

    num=data[0].shape[0]
    valid_size=num//split_fold

    indices = torch.randperm(num)
    train_mapping=indices[valid_size:]
    valid_mapping=indices[:valid_size]


    train_x = torch.index_select(data[0],0,train_mapping)
    valid_x = torch.index_select(data[0],0,valid_mapping)

    train_y = torch.index_select(data[1],0,train_mapping)
    valid_y = torch.index_select(data[1],0,valid_mapping)


    return (train_x,train_y), (valid_x,valid_y)


def prepare_test(filepath, src, dest):
    from shutil import copy

    classes_images = defaultdict(list)
    with open(filepath, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1] + '.jpg')

    for food in classes_images.keys():
        print("\nCopying images into ",food)
        if not os.path.exists(os.path.join(dest,food)):
            os.makedirs(os.path.join(dest,food))
        for i in classes_images[food]:
            try:
                copy(os.path.join(src,food,i), os.path.join(dest,food,i))
            except:
                print("{} not found".format(os.path.join(src,food,i)))
    print("Copying Done!")


def prepare_data(filepath, src, dest):
    from shutil import copy

    classes_images = defaultdict(list)
    with open(filepath, 'r') as txt:
        paths = [read.split("\t")[0].strip() for read in txt.readlines()[1:]]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1])

    for food in classes_images.keys():
        print("\nCopying images into ",food)
        if not os.path.exists(os.path.join(dest,food)):
            os.makedirs(os.path.join(dest,food))
        for i in classes_images[food]:
            try:
                copy(os.path.join(src,food,i), os.path.join(dest,food,i))
            except:
                print("{} not found".format(os.path.join(src,food,i)))

    print("Copying Done!")


def prepare_train(filepath, src, dest):
    from shutil import copy

    classes_images = defaultdict(list)
    with open(filepath, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()[1:]]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1])

    for food in classes_images.keys():
        print("\nCopying images into ",food)
        if not os.path.exists(os.path.join(dest,food)):
            os.makedirs(os.path.join(dest,food))
        for i in classes_images[food]:
            try:
                copy(os.path.join(src,food,i), os.path.join(dest,food,i))
            except:
                print("{} not found".format(os.path.join(src,food,i)))

    print("Copying Done!")


def temperature_scale(logits, temperature):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    temperature = temperature.unsqueeze(0).expand(logits.size(0), logits.size(1))
    return logits / temperature


# This function probably should live outside of this class, but whatever
def get_optimal_temperature(model, valid_loader, device):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    nll_criterion = nn.CrossEntropyLoss().to(device)
    ece_criterion = _ECELoss().to(device)
    temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch_idx, examples in enumerate(valid_loader):
            input, label = examples[0], examples[1]
            input = input.to(device)
            logits = model(input)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

    # Calculate NLL and ECE before temperature scaling
    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temperature_scale(logits, temperature), labels)
        loss.backward()
        return loss
    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_nll = nll_criterion(temperature_scale(logits, temperature), labels).item()
    after_temperature_ece = ece_criterion(temperature_scale(logits, temperature), labels).item()
    print('Optimal temperature: %.3f' % temperature.item())
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    return temperature.item()
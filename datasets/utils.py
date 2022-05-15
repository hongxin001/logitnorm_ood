import torchvision.transforms as trn
import torchvision.datasets as dset
from datasets.cifar import CIFAR10, CIFAR100
import datasets.svhn_loader as svhn


def build_dataset(dataset, mode="train"):

    mean = (0.492, 0.482, 0.446)
    std = (0.247, 0.244, 0.262)

    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                       trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])


    if dataset == 'cifar10':
        if mode == "train":
            data = CIFAR10(root='./data/',
                                    download=True,
                                    dataset_type="train",
                                    transform=train_transform
                                    )
        else:
            data = CIFAR10(root='./data/',
                                   download=True,
                                   dataset_type="test",
                                   transform=test_transform
                                   )
        num_classes = 10
    elif dataset == 'cifar100':
        if mode == "train":
            data = CIFAR100(root='./data/',
                                     download=True,
                                     dataset_type="train",
                                     transform=train_transform
                                     )
        else:
            data = CIFAR100(root='./data/',
                                    download=True,
                                    dataset_type="test",
                                    transform=test_transform
                                    )
        num_classes = 100
    elif dataset == "Textures":
        data = dset.ImageFolder(root="/data/ood_test/dtd/images",
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "SVHN":
        if mode == "train":
            data = svhn.SVHN(root='/data/ood_test/svhn/', split="train",
                             transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
                             download=False)
        else:
            data = svhn.SVHN(root='/data/ood_test/svhn/', split="test",
                             transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
                             download=True)
        num_classes = 10

    elif dataset == "Places365":
        data = dset.ImageFolder(root="/data/ood_test/places365/test_subset",
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "LSUN-C":
        data = dset.ImageFolder(root="/data/ood_test/LSUN_C",
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "LSUN-R":
        data = dset.ImageFolder(root="/data/ood_test/LSUN_R",
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "iSUN":
        data = dset.ImageFolder(root="/data/ood_test/iSUN",
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    return data, num_classes




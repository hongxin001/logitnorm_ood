import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from common.utils import check_integrity,download_url

class SVHN(data.Dataset):
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"],
        'train_and_extra': [
                ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                 "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
                ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                 "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]]}

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test" '
                             'or split="train_and_extra" ')

        if self.split == "train_and_extra":
            self.url = self.split_list[split][0][0]
            self.filename = self.split_list[split][0][1]
            self.file_md5 = self.split_list[split][0][2]
        else:
            self.url = self.split_list[split][0]
            self.filename = self.split_list[split][1]
            self.file_md5 = self.split_list[split][2]

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(root, self.filename))

        if self.split == "test":
            self.data = loaded_mat['X']
            self.targets = loaded_mat['y']
            # Note label 10 == 0 so modulo operator required
            self.targets = (self.targets % 10).squeeze()    # convert to zero-based indexing
            self.data = np.transpose(self.data, (3, 2, 0, 1))
        else:
            self.data = loaded_mat['X']
            self.targets = loaded_mat['y']

            if self.split == "train_and_extra":
                extra_filename = self.split_list[split][1][1]
                loaded_mat = sio.loadmat(os.path.join(root, extra_filename))
                self.data = np.concatenate([self.data,
                                                  loaded_mat['X']], axis=3)
                self.targets = np.vstack((self.targets,
                                               loaded_mat['y']))
            # Note label 10 == 0 so modulo operator required
            self.targets = (self.targets % 10).squeeze()    # convert to zero-based indexing
            self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        if self.split == "test":
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        if self.split == "train_and_extra":
            md5 = self.split_list[self.split][0][2]
            fpath = os.path.join(root, self.filename)
            train_integrity = check_integrity(fpath, md5)
            extra_filename = self.split_list[self.split][1][1]
            md5 = self.split_list[self.split][1][2]
            fpath = os.path.join(root, extra_filename)
            return check_integrity(fpath, md5) and train_integrity
        else:
            md5 = self.split_list[self.split][2]
            fpath = os.path.join(root, self.filename)
            return check_integrity(fpath, md5)

    def download(self):
        if self.split == "train_and_extra":
            md5 = self.split_list[self.split][0][2]
            download_url(self.url, self.root, self.filename, md5)
            extra_filename = self.split_list[self.split][1][1]
            md5 = self.split_list[self.split][1][2]
            download_url(self.url, self.root, extra_filename, md5)
        else:
            md5 = self.split_list[self.split][2]
            download_url(self.url, self.root, self.filename, md5)

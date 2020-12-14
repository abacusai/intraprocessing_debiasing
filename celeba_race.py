import sys
import os
import torch
import torchvision
import numpy as np
from torchvision.datasets import CelebA
from torch.utils.data import Subset

fitz_light = np.load(os.path.expanduser('celebrace/fitz_light.npy'))
fitz_dark = np.load(os.path.expanduser('celebrace/fitz_dark.npy'))


class CelebRace(CelebA):

    def __getitem__(self, index):

        X, target = super().__getitem__(index)
        ind = int(self.filename[index].split('.')[0])

        augment = torch.tensor([fitz_light[ind-1] > .501,
                                fitz_dark[ind-1] > .501,
                                ind,
                                1-target[20]], dtype=torch.long)

        return X, torch.cat((target, augment))


def unambiguous(dataset, split='train', thresh=.7):
    # return only the images which were predicted fitz_light or fitz_dark by >70%

    if split == 'train':
        n = 162770
    else:
        n = 19962
    unambiguous_indices = [i for i in range(n) if (fitz_light[i] > thresh or fitz_dark[i] > thresh)]

    return Subset(dataset, unambiguous_indices)

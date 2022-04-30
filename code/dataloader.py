import torch
import numpy as np
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import h5py
import cv2

import utils
from transforms3d import *

class BrainData(Dataset):
    def __init__(self, train_im_path, mode='train', aug=False):
        self.mode = mode
        self.train_im = h5py.File(train_im_path, 'r')
        self.moving_transform = train_transform(aug=aug)
        self.transform = to_tensor()

        print('{} dataset length: {}'.format(mode, self.__len__()))

    def __len__(self):
        size = len(list(self.train_im.keys()))//4

        return size

    def __getitem__(self, idx):

        moving_name = 'moving_{}'.format(idx)
        moving = np.array(self.train_im[moving_name])
        fixed_name = 'fixed_{}'.format(idx)
        fixed = np.array(self.train_im[fixed_name])

        fixed = self.transform(fixed)
        moving = self.transform(moving)
        mseg_name = 'moving_seg_{}'.format(idx)
        moving_seg = utils.tensor(np.array(self.train_im[mseg_name])).unsqueeze(0)
        fseg_name = 'fixed_seg_{}'.format(idx)
        fixed_seg = utils.tensor(np.array(self.train_im[fseg_name])).unsqueeze(0)

        return {'fixed':fixed, 'moving':moving, 'fixed_seg':fixed_seg, 'moving_seg':moving_seg}

    def getitem(self, idx):
        return self.__getitem__(idx)


def train_transform(affine=False, aug=False):
    transforms = []
    if aug:
        transforms.append(Bspline3D(num=2))

    return T.Compose(transforms)

def to_tensor(resize=128):
    transforms = []
    # transforms.append(Resize3D(resize))
    transforms.append(ToTensor3D())
    # transforms.append(T.Normalize(mean=(0.5), std=(0.5)))
    return T.Compose(transforms)











import torch
import numpy as np
import torchvision as tv
from PIL import Image
from torchvision.transforms import functional as TF

import utils


def toPILImage3D(img):
    ims = []
    for im in img:
        pim = Image.fromarray(im, mode='L')
        ims.append(pim)

    return ims


class ColorJitter3D:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        brightness = np.random.uniform(max(0, 1-self.brightness), max(0, 1+self.brightness))
        contrast = np.random.uniform(max(0, 1-self.contrast), max(0, 1+self.contrast))
        saturation = np.random.uniform(max(0, 1-self.saturation), max(0, 1+self.saturation))
        hue = np.random.uniform(-self.hue, self.hue)

        ims = []
        for im in img:
            im = TF.adjust_brightness(im, brightness)
            im = TF.adjust_contrast(im, contrast)
            im = TF.adjust_saturation(im, saturation)
            im = TF.adjust_hue(im, hue)
            ims.append(im)
        return ims

class RandomAffine3D:
    def __init__(self, degrees=10., translate=(0.1, 0.1), scale=(0.95, 1.05)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def __call__(self, img):
        size = img[0].size()
        degrees = np.random.uniform(-self.degrees, self.degrees)
        max_dx = size[0]*self.translate[0]
        max_dy = size[1]*self.translate[1]
        translate = (np.random.uniform(-max_dx, max_dx),
                     np.random.uniform(-max_dy, max_dy))
        scale = np.random.uniform(self.scale[0], self.scale[1])

        ims = []
        for im in img:
            im = TF.affine(im, degrees, translate, scale)
            ims.append(imgs)

        return ims


class Resize3D:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size, size)
        self.size = size

    def __call__(self, img):
        numpy_im = utils.resize(img, self.size)

        return numpy_im


class ToTensor3D:
    def __init__(self):
        pass

    def __call__(self, img):
        im = utils.tensor(img) / 255.0
        # print(TF.to_tensor(img[0]).size())
        return im.unsqueeze(0)


class Bspline3D(object):
    def __init__(self, mesh_size=(5, 5, 5), level=4, num=1):
        self.mesh_size = mesh_size
        self.level = level
        self.num = num

    def __call__(self, img):
        # img = np.array([np.array(im) for im in img])

        random_n = np.random.randint(self.num)
        for _ in range(random_n):
            img = utils.perturb(img, self.mesh_size, self.level)

        return img



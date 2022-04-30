import torch
import torchvision.utils as vutils
import SimpleITK as sitk
import numpy as np
import cv2
import copy
from scipy import misc
from random import shuffle
from config import Config as cfg
from networks import SpatialTransformer
import utils

idx = 64

class Env(object):
    """
    The environment must specify a root directory including paired files in CT and MR respectively
    The virtual_label is used to assist the registration learning process
    """
    def __init__(self, generator, stn, seg_stn, use_seg=False, device=None):
        # self.generator = self._generator(dataset)
        self.generator = generator
        self.stn = stn
        self.seg_stn = seg_stn
        self.use_seg = use_seg
        self.device = device
        # self.labels = utils.brain_labels()
        self.reset()


    def _generator(self, datasets):
        indices = np.arange(len(datasets))
        self.epoch = 0
        while True:
            np.random.shuffle(indices)
            self.epoch += 1
            for idx in indices:
                items = datasets.getitem(idx)
                yield items

    def reset(self):

        items = next(self.generator)
        # print(items)

        self.tensor_fixed = utils.tensor(items['fixed']).to(self.device)
        self.tensor_moving = utils.tensor(items['moving']).to(self.device)
        self.tensor_moved = copy.deepcopy(self.tensor_moving)

        if self.use_seg:
            self.fixed_seg = items['fixed_seg']
            # self.labels = np.unique(self.fixed_seg)
            self.moving_seg = utils.tensor(items['moving_seg']).to(self.device)

            # self.moving_seg = []
            # for i, lab in enumerate(self.labels):
            #     m_seg = (m_segs == lab)*255.
            #     self.moving_seg.append(m_seg)

            self.moved_seg = copy.deepcopy(self.moving_seg)

        self.prev_score = self.score()
        self.field = None
        self.global_step = 1
        return self.state(), self.prev_score

    def state(self):
        return torch.cat([self.tensor_fixed, self.tensor_moved], dim=1).to(self.device)

    def score(self):
        if self.use_seg:
            # dices = []
            # for i, lab in enumerate(self.labels):
            #     f_seg = (self.fixed_seg == lab).astype(np.uint8)
            #     m_seg = utils.numpy_im(self.moved_seg[i] > 0, 1, device=self.device)

            #     dice = utils.dice_(f_seg, m_seg, [1])
            #     dices.append(dice)
            dices = utils.dice(self.fixed_seg[0, 0]>0,
                utils.numpy_im(self.moved_seg, 1, device=self.device)>0)
            return np.mean(dices)
        else:
            return utils.ncc_tensor_score(self.tensor_moved, self.tensor_fixed).cpu().item()

    def done(self):
        return self.score() > cfg.SCORE_THRESHOLD

    # latent is a tensor representing the displacement field
    def step(self, field):
        if self.field is None:
            self.field = field
        else:
            self.field = self.stn(self.field, field) + field
            self.global_step += 1

        self.step_field = field

        self.tensor_moved = self.stn(self.tensor_moving, self.field)
        self.moved_seg = self.seg_stn(self.moving_seg, self.field)
        # if self.use_seg:
        #     for i, lab in enumerate(self.labels):
        #         self.moved_seg[i] = self.stn(self.moving_seg[i], self.field)

        image_score = self.score()

        reward = (image_score - self.prev_score)*100
        self.prev_score = image_score
        if self.done():
            reward += 10

        return reward, self.state(), self.done(), image_score

    def save_init(self, dir=cfg.PROCESS_PATH):
        vutils.save_image(self.tensor_fixed[:, :, :, idx, :].data, dir+'/fixed.bmp', normalize=True)
        vutils.save_image(self.tensor_moving[:, :, :, idx, :].data, dir+'/moving.bmp', normalize=True)

    def save_process(self, index, dir=cfg.PROCESS_PATH):
        if index % 3 == 0:
            # tmp_field = utils.render_flow(utils.numpy_im(self.field, 1., self.device)[:, :, idx, :])
            # cv2.imwrite('{}/field-{}.png'.format(dir, index), tmp_field)
            vutils.save_image(self.tensor_moved[:, :, :, idx, :].data, dir+'/moved-{}.bmp'.format(index), normalize=True)



























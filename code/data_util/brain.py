import json
import copy
import numpy as np
import collections

import utils
from .liver import FileManager
from .liver import Dataset as BaseDataset

# import ants


# def segmentation(numpy_im):
#     # print('segmentation...')
#     ants_im = ants.from_numpy(numpy_im).astype('float32')
#     mask = ants.get_mask(ants_im, 10, 255, 0)
#     seg = ants.kmeans_segmentation(ants_im, k=3, kmask=mask, mrf=0.2)
#     return seg['segmentation'].numpy()

class Dataset(BaseDataset):
    def __init__(self, split_path, paired=False, task=None, batch_size=None):
        with open(split_path, 'r') as f:
            config = json.load(f)
        self.files = FileManager(config['files'])
        self.subset = {}

        for k, v in config['subsets'].items():
            self.subset[k] = {}
            for entry in v:
                self.subset[k][entry] = self.files[entry]

        self.paired = paired

        def convert_int(key):
            try:
                return int(key)
            except ValueError as e:
                return key
        self.schemes = dict([(convert_int(k), v)
                             for k, v in config['schemes'].items()])

        for k, v in self.subset.items():
            print('Number of data in {} is {}'.format(k, len(v)))

        self.task = task
        if self.task is None:
            self.task = config.get("task", "registration")
        if not isinstance(self.task, list):
            self.task = [self.task]

        self.image_size = config.get("image_size", [32, 32]) ###
        # self.image_size = [64, 64, 64] ###
        self.segmentation_class_value = config.get(
            'segmentation_class_value', None)

        if 'atlas' in config:
            self.atlas = self.files[config['atlas']]
        else:
            self.atlas = None

        if paired:
            self.atlas = None

        self.batch_size = batch_size
        self.fixed_seg = None

    def center_crop(self, volume):
        slices = [slice((os - ts) // 2, (os - ts) // 2 + ts) if ts < os else slice(None, None)
                  for ts, os in zip(self.image_size, volume.shape)]
        volume = volume[slices]

        ret = np.zeros(self.image_size, dtype=volume.dtype)
        slices = [slice((ts - os) // 2, (ts - os) // 2 + os) if ts > os else slice(None, None)
                  for ts, os in zip(self.image_size, volume.shape)]
        ret[slices] = volume

        return ret

    @staticmethod
    def generate_atlas(atlas, sets, loop=False):
        sets = copy.copy(sets)
        while True:
            if loop:
                np.random.shuffle(sets)
            for d in sets:
                yield atlas, d
            if not loop:
                break

    def generator(self, subset, batch_size=None, loop=False, aug=False):
        if batch_size is None:
            batch_size = self.batch_size
        scheme = self.schemes[subset]
        if 'registration' in self.task:
            if self.atlas is not None:
                generators, fractions = zip(*[(self.generate_atlas(self.atlas, list(
                    self.subset[k].values()), loop), fraction) for k, fraction in scheme.items()])
            else:
                generators, fractions = zip(
                    *[(self.generate_pairs(list(self.subset[k].values()), loop), fraction) for k, fraction in scheme.items()])

            while True:
                i = 0
                flag = True
                ret = dict()
                for gen in generators:
                    try:
                        while True:
                            d1, d2 = next(gen)
                            break
                    except StopIteration:
                        flag = False
                        break

                    fixed = np.array(d1['volume'])
                    moving = np.array(d2['volume'])

                    if aug:
                        moving = utils.perturb(moving, tmp_level=5)

                    ret['fixed'] = utils.tensor(fixed/255.)[None,None,...]
                    ret['moving'] = utils.tensor(moving/255.)[None,None,...]

                    if 'segmentation' in d1:
                        ret['fixed_seg'] = np.array(d1['segmentation'])[None,None,...]

                    if 'segmentation' in d2:
                        ret['moving_seg'] = np.array(d2['segmentation'])[None,None,...]

                    # if subset == 1:
                    #     if self.fixed_seg is None:
                    #         self.fixed_seg = utils.segmentation(fixed)[None,None,...]
                    #     ret['fixed_seg'] = copy.deepcopy(self.fixed_seg)
                    #     ret['moving_seg'] = utils.segmentation(moving)[None,None,...]

                    i += 1

                if flag:
                    # print(ret)
                    yield ret
                else:
                    # print(ret)
                    # yield ret
                    break



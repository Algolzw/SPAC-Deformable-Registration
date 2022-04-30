import numpy as np
import json
import os
import h5py
import ants
import utils

from .data import Split


# import ants


# def segmentation(numpy_im):
#     # print('segmentation...')
#     ants_im = ants.from_numpy(numpy_im).astype('float32')
#     mask = ants.get_mask(ants_im, 10, 255, 0)
#     seg = ants.kmeans_segmentation(ants_im, k=3, kmask=mask, mrf=0.2)
#     return seg['segmentation'].numpy()

def affine_with_label(moving, fixed, m_seg, f_seg):
    fixed = ants.from_numpy(fixed)
    moving = ants.from_numpy(moving)
    m_seg = ants.from_numpy(m_seg)
    f_seg = ants.from_numpy(f_seg)

    result = ants.registration(fixed, moving,
        'TRSAA', aff_metric='meansquares', aff_iterations=[1000, 1000, 1000, 5])
    warped = result['warpedmovout']
    tranform = result['fwdtransforms']
    warped_seg = ants.apply_transforms(fixed=f_seg, moving=m_seg,
        transformlist=tranform, interpolator='nearestNeighbor')

    return warped.numpy(), warped_seg.numpy()


def get_range(imgs):
    r = np.any(imgs.reshape(imgs.shape[0], -1), axis=-1).nonzero()
    return np.min(r), np.max(r)


class Hdf5Reader:
    def __init__(self, path):
        try:
            self.file = h5py.File(path, "r")
        except Exception:
            print('{} not found!'.format(path))
            self.file = None

    def __getitem__(self, key):
        data = {'id': key}
        if self.file is None:
            return data
        group = self.file[key]
        for k in group:
            data[k] = group[k]
        return data


class FileManager:
    def __init__(self, files):
        self.files = {}
        for k, v in files.items():
            self.files[k] = Hdf5Reader(v["path"])

    def __getitem__(self, key):
        p = key.find('/')
        if key[:p] in self.files:
            ret = self.files[key[:p]][key[p+1:]]
            ret['id'] = key.replace('/', '_')
            return ret
        elif '/' in self.files:
            ret = self.files['/'][key]
            ret['id'] = key.replace('/', '_')
            return ret
        else:
            raise KeyError('{} not found'.format(key))


class Dataset:
    def __init__(self, split_path, affine=False, mask=False, paired=False, task=None, batch_size=None):
        with open(split_path, 'r') as f:
            config = json.load(f)
        self.files = FileManager(config['files'])
        self.subset = {}

        for k, v in config['subsets'].items():
            self.subset[k] = {}
            for entry in v:
                self.subset[k][entry] = self.files[entry]

        self.paired = paired
        self.affine = affine

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

        self.batch_size = batch_size

    def get_pairs_adj(self, data):
        pairs = []
        d1 = None
        for d2 in data:
            if d1 is None:
                d1 = d2
            else:
                pairs.append((d1, d2))
                pairs.append((d2, d1))
                d1 = None
        return pairs

    def get_pairs(self, data, ordered=True):
        pairs = []
        for i, d1 in enumerate(data):
            for j, d2 in enumerate(data):
                if i != j:
                    if ordered or i < j:
                        pairs.append((d1, d2))
        return pairs

    def generate_pairs(self, arr, loop=False):
        if self.paired:
            sets = self.get_pairs_adj(arr)
        else:
            sets = self.get_pairs(arr, ordered=True)

        while True:
            if loop:
                np.random.shuffle(sets)
            for d1, d2 in sets:
                yield (d1, d2)
            if not loop:
                break

    def generator(self, subset, batch_size=None, loop=False):
        if batch_size is None:
            batch_size = self.batch_size
        valid_mask = np.ones([6], dtype=np.bool)
        scheme = self.schemes[subset]
        if 'registration' in self.task:
            generators = [(self.generate_pairs(list(self.subset[k].values()), loop))
                          for k, fraction in scheme.items()]
            fractions = [int(np.round(fraction * batch_size))
                         for k, fraction in scheme.items()]

            while True:
                ret = dict()
                # ret['voxel1'] = np.zeros(
                #     (batch_size, 128, 128, 128, 1), dtype=np.float32)
                # ret['voxel2'] = np.zeros(
                #     (batch_size, 128, 128, 128, 1), dtype=np.float32)
                # ret['seg1'] = np.zeros(
                #     (batch_size, 128, 128, 128, 1), dtype=np.float32)
                # ret['seg2'] = np.zeros(
                #     (batch_size, 128, 128, 128, 1), dtype=np.float32)
                # ret['point1'] = np.ones(
                #     (batch_size, np.sum(valid_mask), 3), dtype=np.float32) * (-1)
                # ret['point2'] = np.ones(
                #     (batch_size, np.sum(valid_mask), 3), dtype=np.float32) * (-1)
                # ret['id1'] = np.empty((batch_size), dtype='<U40')
                # ret['id2'] = np.empty((batch_size), dtype='<U40')

                i = 0
                flag = True
                nums = fractions
                for gen, num in zip(generators, nums):
                    assert not self.paired or num % 2 == 0
                    for t in range(num):
                        try:
                            while True:
                                d1, d2 = next(gen)
                                break
                        except StopIteration:
                            flag = False
                            break

                        fixed = np.array(d1['volume'])
                        moving = np.array(d2['volume'])

                        f_seg = np.array(d1['segmentation'])
                        m_seg = np.array(d2['segmentation'])

                        if self.affine:
                            moving, m_seg = affine_with_label(moving, fixed, m_seg, f_seg)

                        ret['fixed'] = utils.tensor(fixed/255.)[None,None,...]
                        ret['moving'] = utils.tensor(moving/255.)[None,None,...]

                        if 'segmentation' in d1:
                            ret['fixed_seg'] = f_seg[None,None,...]
                            # ret['fixed_seg'] = utils.segmentation(fixed)[None,None,...]

                        if 'segmentation' in d2:
                            ret['moving_seg'] = m_seg[None,None,...]
                            # ret['moving_seg'] = utils.segmentation(moving)[None,None,...]

                        i += 1

                if flag:
                    assert i == batch_size
                    yield ret
                else:
                    yield ret
                    break

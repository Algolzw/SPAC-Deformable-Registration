import torch
import random
import numpy as np
import copy

import utils
import data_util.brain
import data_util.liver
from data_util.data import Split
from dataloader import BrainData
from config import Config as cfg
from brain import SPAC
from env import Env
from agent import Agent
from summary import Summary
from networks import *

# os.environ['CUDA_VISIBLE_DEVICES'] = pa.GPU_ID

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(cfg.GPU_ID)
else:
    device = torch.device('cpu')

# device = torch.device('cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # cpu\gpu 结果一致

if __name__ == "__main__":
    setup_seed(cfg.SEED)
    utils.mkdir(cfg.LOG_DIR)
    utils.remkdir(cfg.PROCESS_PATH)
    utils.mkdir(cfg.MODEL_PATH)

    summary = Summary(cfg.LOG_DIR)

    #######################################
    stn = SpatialTransformer(cfg.HEIGHT, 'bilinear').to(device) # nearest
    seg_stn = SpatialTransformer(cfg.HEIGHT, mode='nearest').to(device) # nearest

    # datasets = BrainData(cfg.TRAIN_DATA, cfg.TRAIN_SEG_DATA, cfg.ATLAS, use_seg=cfg.USE_SEG, mode='train', aug=cfg.BSPLINE_AUG)

    Dataset = eval('data_util.{}.Dataset'.format(cfg.IMAGE_TYPE))
    dataset = Dataset(split_path='datasets/%s.json' % cfg.IMAGE_TYPE, paired=False, affine=True)
    generator = dataset.generator(cfg.DATA_TYPE, batch_size=1, loop=True)


    brain = SPAC(stn, device)
    if cfg.PRE_TRAINED:
        print('pretrain models')
        brain.load_model('actor', cfg.ACTOR_MODEL)
        brain.load_model('decoder', cfg.DECODER_MODEL_RL)
        brain.load_model('critic1', cfg.CRITIC1_MODEL)
        brain.load_model('critic2', cfg.CRITIC2_MODEL)

    env = Env(generator, stn, seg_stn, cfg.USE_SEG, device)
    agent = Agent(brain, env, summary, device=device)

    agent.run()







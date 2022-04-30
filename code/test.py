import torch
import random
import numpy as np
import copy
import cv2
import time

import utils
from dataloader import BrainData
from config import Config as cfg
from brain import SPAC
from env import Env
from summary import Summary
from networks import *

# from thop import profile
# from thop import clever_format

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
    idx = 100
    setup_seed(cfg.SEED)
    utils.remkdir(cfg.TEST_PATH)

    #######################################
    # stn = SpatialTransformer(device).to(device)
    stn = SpatialTransformer(cfg.HEIGHT).to(device)
    seg_stn = SpatialTransformer(cfg.HEIGHT, mode='nearest').to(device)

    # test_data = BrainData(cfg.TEST_DATA, mode='test', size=cfg.HEIGHT, affine=False)
    test_loader = BrainData(cfg.TEST_DATA, mode='test')
    # test_loader = test_data.generator()
    brain = SPAC(stn, device)
    brain.load_model('actor', cfg.ACTOR_MODEL)
    brain.load_model('decoder', cfg.DECODER_MODEL_RL)
    brain.load_model('critic1', cfg.CRITIC1_MODEL)
    brain.load_model('critic2', cfg.CRITIC2_MODEL)

    brain.eval(brain.actor)
    brain.eval(brain.critic1)
    brain.eval(brain.critic2)
    brain.eval(brain.decoder)


    # model = VAE(cfg).to(device)
    # input = torch.randn(1, 2, 128, 128, 128).cuda()
    # macs, params = profile(model, inputs=(input, ))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('flops: {}, params: {}'.format(macs, params))

    dices = []
    times = []
    # grid = utils.virtual_grid(cfg.HEIGHT, tensor=True).unsqueeze(0).to(device)
    # print(grid.shape)
    print(len(test_loader))
    latents = []
    labs = []
    jacobian_dets = []
    for i, item in enumerate(test_loader):
        fixed = item['fixed']
        fixed_seg = item['fixed_seg']
        moving = item['moving']
        moving_seg = item['moving_seg']

        if i % 1 == 0:
            cv2.imwrite('{}/{}-fixed.bmp'.format(cfg.TEST_PATH, i), utils.numpy_im(fixed)[:, :, idx])
            cv2.imwrite('{}/{}-moving.bmp'.format(cfg.TEST_PATH, i), utils.numpy_im(moving)[:, :, idx])
            # cv2.imwrite('{}/{}-fixed_seg.bmp'.format(cfg.TEST_PATH, i), utils.numpy_im(fixed_seg>0, 255.)[:, :, idx])
            # cv2.imwrite('{}/{}-moving_seg.bmp'.format(cfg.TEST_PATH, i), utils.numpy_im(moving_seg>0, 255.)[:, :, idx])

        fixed_seg = utils.numpy_im(fixed_seg, 1)
        moving_seg = moving_seg.to(device)[None,...]
        fixed = fixed.to(device).unsqueeze(0)
        moving = moving.to(device).unsqueeze(0)

        moved = copy.deepcopy(moving)

        pred = None
        best_pred = None
        best_value = 0
        best_step = 0
        step = 0
        tic = time.time()

        while step < 20:
            state = torch.cat([fixed, moved], dim=1)
            latent, flow = brain.choose_action(state, test=True)
            latents.append(latent.cpu().numpy())
            labs.append(i)
            pred = flow if pred is None else stn(pred, flow) + flow

            moved = stn(moving, pred)

            step += 1


        toc = time.time()
        warped_im = utils.numpy_im(stn(moving, pred), device=device)
        warped_seg = utils.numpy_im(seg_stn(moving_seg, pred), 1, device)#.astype(np.uint8)
        # moving_seg_numpy = utils.numpy_im(moving_seg, 1, device=device)
        # warped_grid = utils.numpy_im(seg_stn(grid, pred), device=device).astype(np.uint8)

        # scores = []
        # warped_seg = cv2.blur(warped_seg, (3, 3))
        labels = np.unique(fixed_seg)[1:]

        scores = utils.dice(fixed_seg>0, warped_seg>0, [1])
        score = np.mean(scores)
        dices.append(score)

        times.append(toc-tic)

        flow = utils.numpy(pred.squeeze(), device=device)
        jacobian_det = np.mean(utils.jacobian_determinant(np.transpose(flow, (1, 2, 3, 0))))
        jacobian_dets.append(jacobian_det)

        if i % 1 == 0:
            flow = utils.render_flow(flow[:, :, :, idx])
            # cv2.imwrite('result/{}-flow.png'.format(i), flow)
            cv2.imwrite('{}/{}-pred_img.bmp'.format(cfg.TEST_PATH, i), warped_im[:, :, idx])

            # cv2.imwrite('result/{}-pred_seg.bmp'.format(i), (warped_seg[:, :, idx]>0)*255)

            # vis_seg = utils.visualize(fixed_seg[:, :, idx], warped_seg[:, :, idx])
            # cv2.imwrite('result/{}-vis.png'.format(i), vis_seg*255)

            vis_seg = utils.render_image_with_mask(utils.numpy_im(fixed, device=device)[:, :, idx], warped_seg[:, :, idx], color=1)
            cv2.imwrite('{}/{}-vis_pre.png'.format(cfg.TEST_PATH, i), vis_seg)

            vis_seg = utils.render_image_with_mask(utils.numpy_im(fixed, device=device)[:, :, idx], fixed_seg[:, :, idx], color=0)
            cv2.imwrite('{}/{}-vis_gt.png'.format(cfg.TEST_PATH, i), vis_seg)

        ndices = np.array(dices)
        print('data-{}: avg time: {:.4f}, dice: {:.4f}, jacobian_det: {:.4f} final dice: {:.4f}({:.4f})'
            .format(i, np.mean(times), score, jacobian_det, np.mean(ndices), np.std(ndices)))
        if i == 16:
            break
    # latents = np.concatenate(latents, axis=0)
    # utils.computeTSNEProjectionOfLatentSpace(latents, labs, True, i, step)

    print('avg time: {:.4f}, final dice: {:.4f}({:.4f}), 50%: {:.4f}, 90%: {:.4f}'
        .format(np.mean(times),
            np.mean(ndices),
            np.std(ndices),
            np.median(ndices),
            np.percentile(ndices, 90)))

    print('final jacobian determinant: {:.4f}({:.4f})'.format(np.mean(jacobian_dets), np.std(jacobian_dets)))















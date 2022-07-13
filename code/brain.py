import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

import utils
from networks import *
from config import Config as cfg

class SPAC:
    def __init__(self, stn, device=None):
        self.stn = stn
        self.device = device
        self.target_entropy = -16

        self.log_alpha = torch.tensor(-1.0).to(device).detach().requires_grad_(True)

        self.planner = Encoder(nc=cfg.STATE_CHANNEL, nf=cfg.NF, bottle=cfg.BOTTLE).to(device)
        self.actor = Decoder(nf=cfg.NF, bottle=cfg.BOTTLE).to(device)

        self.critic1 = Critic(nc=cfg.STATE_CHANNEL, nf=cfg.NF, bottle=cfg.BOTTLE).to(device)
        self.critic2 = Critic(nc=cfg.STATE_CHANNEL, nf=cfg.NF, bottle=cfg.BOTTLE).to(device)

        self.critic1_target = Critic(nc=cfg.STATE_CHANNEL, nf=cfg.NF, bottle=cfg.BOTTLE).to(device)
        self.critic2_target = Critic(nc=cfg.STATE_CHANNEL, nf=cfg.NF, bottle=cfg.BOTTLE).to(device)

        self.eval(self.critic1_target)
        self.eval(self.critic2_target)

        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.LEARNING_RATE, betas=(0.5, 0.9))
        self.planner_optim = torch.optim.Adam(self.planner.parameters(), lr=cfg.LEARNING_RATE/5, betas=(0.5, 0.9))
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.9))
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=cfg.LEARNING_RATE/5, betas=(0.5, 0.9))
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=cfg.LEARNING_RATE/5, betas=(0.5, 0.9))

        self.hard_update(self.critic1_target, self.critic1)
        self.hard_update(self.critic2_target, self.critic2)

    @property
    def alpha(self):
        # if cfg.FIXED_ALPHA is not None:
        #     return cfg.FIXED_ALPHA
        return torch.exp(self.log_alpha)

    def get_aff_flow(self, latent, xs, ys, zs, return_A=False):
        latent1 = latent[:, :-12]
        W = latent[:, -12:-3].reshape(-1, 3, 3)
        b = latent[:, -3:].reshape(-1, 3)

        I = torch.eye(3).to(self.device)
        A = W + I

        flow = utils.aff_flow(W, b, xs, ys, zs)
        if return_A:
            return latent1, flow, A
        return latent1, flow


    def get_value(self, state, action):
        self.critic1.eval()
        self.critic2.eval()
        v1 = self.critic1(state, action)
        v2 = self.critic2(state, action)
        return torch.min(v1, v2).item()

    def choose_action(self, state, test=False):
        self.planner.eval()
        self.actor.eval()

        dist, enc = self.planner(state)

        if test:
            latent = dist.mean
        else:
            latent = dist.sample()

        # xs, ys, zs = state.size()[-3:]
        # flow_latent, aff_field = self.get_aff_flow(latent, xs, ys, zs)

        field = self.actor(latent, enc)
        if not test:
            field = field.clamp(-1, 1)
        else:
            field = field.clamp(-3, 3)

        # field = self.stn(aff_field, field) + field

        return latent.detach(), field.detach()

    def optimize(self, update_planner, update_actor, samples):
        """ update_planner: should update planner now?
            samples: s, a, r, s_
        """
        self.planner.train()
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

        loss = self._loss(samples, update_planner, update_actor)

        self.soft_update(self.critic1_target, self.critic1)
        self.soft_update(self.critic2_target, self.critic2)

        return loss


    def _loss(self, samples, update_planner, update_actor):
        s, a, r, s_, done = samples

        s = utils.tensor(s, device=self.device)
        a = utils.tensor(a, device=self.device)
        r = utils.tensor(r[..., None], device=self.device)
        s_ = utils.tensor(s_, device=self.device)
        done = utils.tensor(done[..., None],  device=self.device)

        loss = {}

        ######## critic loss #######
        dist_, enc_ = self.planner(s_)
        a_ = dist_.sample()

        log_pi_next = dist_.log_prob(a_)
        Q_target1_next = self.critic1_target(s_, a_)
        Q_target2_next = self.critic2_target(s_, a_)
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)

        Q_target = r + (cfg.GAMMA * (1-done) * (Q_target_next - self.alpha * log_pi_next.unsqueeze(1)))

        Q1 = self.critic1(s, a)
        Q2 = self.critic2(s, a)

        critic1_loss = F.mse_loss(Q1, Q_target.detach())
        critic2_loss = F.mse_loss(Q2, Q_target.detach())
        loss['critic1'] = critic1_loss
        loss['critic2'] = critic2_loss

        self.critic1_optim.zero_grad()
        loss['critic1'].backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        loss['critic2'].backward()
        self.critic2_optim.step()

        ########## planner loss  ############
        if update_planner:
            dist, enc = self.planner(s)
            action = dist.rsample()
            log_pi = dist.log_prob(action)
            # alpha loss
            alpha_loss = -(self.log_alpha * (log_pi.unsqueeze(1) + self.target_entropy).detach()).mean()
            loss['alpha'] = alpha_loss

            self.alpha_optim.zero_grad()
            loss['alpha'].backward()
            self.alpha_optim.step()

            #############planner loss###############
            planner_Q1 = self.critic1(s, action)
            planner_Q2 = self.critic2(s, action)
            planner_Q = torch.min(planner_Q1, planner_Q2)
            planner_loss = (self.alpha.detach() * log_pi.unsqueeze(1) - planner_Q).mean()
            kl_loss = self.kl_loss(dist)

            loss['planner'] = planner_loss + cfg.ENTROPY_BETA*kl_loss

            self.planner_optim.zero_grad()
            loss['planner'].backward(retain_graph=False)
            self.planner_optim.step()

        ########### resgistration loss ##########
        if update_actor:
            vae_dist, vae_enc = self.planner(s)
            latent = vae_dist.rsample()
            flow = self.actor(latent, vae_enc)
            warped = self.stn(s[:, 1:], flow)
            reg_loss =  utils.multi_scale_ncc_loss(s[:, :1], warped) + 1.*utils.gradient_loss(flow) + 0.01*self.kl_loss(vae_dist)
            loss['reg'] = reg_loss

            self.actor_optim.zero_grad()
            loss['reg'].backward()
            self.actor_optim.step()

        return loss

    @staticmethod
    def soft_update(target, source, tau=0.005):
        """
        Copies the parameters from source network (x) to target network (y) using the below update
        y = TAU*x + (1 - TAU)*y
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    @staticmethod
    def hard_update(target, source):
        """
        Copies the parameters from source network to target network
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

    def kl_loss(self, dist):
        mu = dist.mean
        var = dist.variance
        logvar = torch.log(var)
        loss = 0.5*torch.mean(var+mu.pow(2)-1-logvar)
        return loss

    def eval(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def save_model(self, step, model_path):
        torch.save(self.planner.state_dict(), os.path.join(model_path,'planner_{}.ckpt'.format(step)))
        torch.save(self.critic1.state_dict(), os.path.join(model_path,'critic1_{}.ckpt'.format(step)))
        torch.save(self.critic2.state_dict(), os.path.join(model_path,'critic2_{}.ckpt'.format(step)))
        torch.save(self.actor.state_dict(), os.path.join(cfg.MODEL_PATH,'actor_{}.ckpt'.format(step)))

    def load_model(self, name, model_path):
        eval("self.{}.load_state_dict(torch.load('{}'))".format(name, model_path))

    def load_planner(self, model_path):
        self.planner.load_state_dict(torch.load(model_path,  map_location={'cuda:7': 'cuda:6'}))

    def load_critic(self, critic1_path, critic2_path):
        self.critic1.load_state_dict(torch.load(critic1_path,  map_location={'cuda:7': 'cuda:6'}))
        self.critic2.load_state_dict(torch.load(critic2_path,  map_location={'cuda:7': 'cuda:6'}))

    def load_actor(self, model_path):
        self.actor.load_state_dict(torch.load(model_path,  map_location={'cuda:7': 'cuda:6'}))





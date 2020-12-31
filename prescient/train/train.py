# shared functions and classes, including the model and `run`
# which implements the main pre-training and training loop

import torch
import torch.nn.functional as F
from torch import nn, optim

import numpy as np

from geomloss import SamplesLoss

import tqdm

from collections import OrderedDict
from types import SimpleNamespace
from time import strftime, localtime

import argparse
import itertools
import json
import os
import sys

import sklearn.decomposition

# ---- model

class IntReLU(nn.Module):

    def __init__(self, input_dim):
        super(IntReLU, self).__init__()

    def forward(self, x):
        return torch.max(torch.zeros_like(x), 0.5 * (x**2)) # + self.c)


class AutoGenerator(nn.Module):

    def __init__(self, config):
        super(AutoGenerator, self).__init__()

        self.x_dim = config.x_dim
        self.k_dim = config.k_dim
        self.layers = config.layers

        self.activation = config.activation
        if self.activation == 'relu':
            self.act = nn.LeakyReLU
        elif self.activation == 'softplus':
            self.act = nn.Softplus
        elif self.activation == 'intrelu': # broken, wip
            raise NotImplementedError
        elif self.activation == 'none':
            self.act = None
        else:
            raise NotImplementedError

        self.net_ = []
        for i in range(self.layers):
            # add linear layer
            if i == 0:
                self.net_.append(('linear{}'.format(i+1), nn.Linear(self.x_dim, self.k_dim)))
            else:
                self.net_.append(('linear{}'.format(i+1), nn.Linear(self.k_dim, self.k_dim)))
            # add activation
            if self.activation == 'intrelu':
                raise NotImplementedError
            elif self.activation == 'none':
                pass
            else:
                self.net_.append(('{}{}'.format(self.activation, i+1), self.act()))
        self.net_.append(('linear', nn.Linear(self.k_dim, 1, bias = False)))
        self.net_ = OrderedDict(self.net_)
        self.net = nn.Sequential(self.net_)

        net_params = list(self.net.parameters())
        net_params[-1].data = torch.zeros(net_params[-1].data.shape) # initialize

    def _step(self, x, dt, z):
        sqrtdt = np.sqrt(dt)
        return x + self._drift(x) * dt + z * sqrtdt

    def _pot(self, x):
        return self.net(x)

    def _drift(self, x):
        x_ = x.requires_grad_()
        pot = self._pot(x_)

        drift = torch.autograd.grad(pot, x_, torch.ones_like(pot),
            create_graph = True)[0]
        return drift

# ---- loss

class OTLoss():

    def __init__(self, config, device):

        self.ot_solver = SamplesLoss("sinkhorn", p = 2, blur = config.sinkhorn_blur,
            scaling = config.sinkhorn_scaling, debias = True)
        self.device = device

    def __call__(self, a_i, x_i, b_j, y_j, requires_grad = True):

        a_i = a_i.to(self.device)
        x_i = x_i.to(self.device)
        b_j = b_j.to(self.device)
        y_j = y_j.to(self.device)

        if requires_grad:
            a_i.requires_grad_()
            x_i.requires_grad_()
            b_j.requires_grad_()

        loss_xy = self.ot_solver(a_i, x_i, b_j, y_j)
        return loss_xy

def run(args, init_task):

    # ---- initialize

    device, kwargs = init(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y, w, config = init_task(args)

    # ---- model

    model = AutoGenerator(config)
    print(model)
    model.zero_grad()

    # ---- loss

    if args.loss == 'euclidean':
        loss = OTLoss(config, device)
    else:
        raise NotImplementedError

    torch.save(config.__dict__, config.config_pt)

    if args.pretrain:

        if os.path.exists(config.done_log):

            print(config.done_log, ' exists. Skipping.')

        else:

            model.to(device)

            x_last = x[config.train_t[-1]].to(device) # use the last available training point

            optimizer = optim.SGD(list(model.parameters()), lr = config.pretrain_lr)

            pbar = tqdm.tqdm(range(config.pretrain_epochs))
            for epoch in pbar:

                pp, _ = p_samp(x_last, config.ns)

                dt = config.t / config.pretrain_burnin
                pp, pos_fv, neg_fv = fit_regularizer(x_last, pp,
                    config.pretrain_burnin, dt, config.pretrain_sd,
                    model, device)
                fv_tot = pos_fv + neg_fv

                fv_tot.backward()
                optimizer.step()
                model.zero_grad()

                pbar.set_description('[{}|pretrain] {} {:.3f}'.format(
                    config.out_name, epoch, fv_tot.item()))

            torch.save({
                'model_state_dict': model.state_dict(),
            }, config.pretrain_pt)

    if args.train:

        if os.path.exists(config.done_log):

            print(config.done_log, ' exists. Skipping.')

        else:

            checkpoint = torch.load(config.pretrain_pt)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            optimizer = optim.Adam(list(model.parameters()), lr = config.train_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.9)
            optimizer.zero_grad()

            pbar = tqdm.tqdm(range(config.train_epochs))
            x_last = x[config.train_t[-1]].to(device) # use the last available training point

            # fit on time points

            best_train_loss_xy = np.inf
            log_handle = open(config.train_log, 'w')

            for epoch in pbar:

                losses_xy = []
                config.train_epoch = epoch

                for j in config.train_t:

                    t_cur = j
                    t_prev = config.start_t
                    dat_cur = x[t_cur]
                    dat_prev = x[t_prev]
                    y_cur = y[t_cur]
                    y_prev = y[t_prev]
                    time_elapsed = y_cur - y_prev

                    w_prev = get_weight(w[(y_prev, y_cur)], time_elapsed)

                    x_i, a_i = p_samp(dat_prev, int(dat_prev.shape[0] * args.train_batch),
                        w_prev)
                    x_i = x_i.to(device)
                    num_steps = int(np.round(time_elapsed / config.train_dt))
                    for _ in range(num_steps):
                        z = torch.randn(x_i.shape[0], x_i.shape[1]) * config.train_sd
                        z = z.to(device)
                        x_i = model._step(x_i, dt = config.train_dt, z = z)

                    y_j, b_j = p_samp(dat_cur, int(dat_cur.shape[0] * args.train_batch))

                    loss_xy = loss(a_i, x_i, b_j, y_j)
                    losses_xy.append(loss_xy.item())

                    #[F_i, G_j, dx_i] = torch.autograd.grad( Loss_xy, [a_i, b_j, x_i] )

                    loss_xy.backward()

                train_loss_xy = np.mean(losses_xy)

                # fit regularizer

                if config.train_tau > 0:

                    pp, _ = p_samp(x_last, config.ns)

                    dt = config.t / config.train_burnin
                    pp, pos_fv, neg_fv = fit_regularizer(x_last, pp,
                        config.train_burnin, dt, config.train_sd,
                        model, device)
                    fv_tot = pos_fv + neg_fv
                    fv_tot *= config.train_tau
                    fv_tot.backward()

                # step

                if config.train_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train_clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                # report

                desc = "[{}|train] {}".format(config.out_name, epoch + 1)
                if len(losses_xy) < 10:
                    for l_xy in losses_xy:
                        desc += " {:.6f}".format(l_xy)
                desc += " {:.6f}".format(train_loss_xy)
                desc += " {:.6f}".format(best_train_loss_xy)
                pbar.set_description(desc)
                log_handle.write(desc + '\n')
                log_handle.flush()

                if train_loss_xy < best_train_loss_xy:
                    best_train_loss_xy = train_loss_xy

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': config.train_epoch + 1,
                    }, config.train_pt.format('best'))

                # save model every x epochs

                if (config.train_epoch + 1) % config.save == 0:
                    epoch_ = str(config.train_epoch + 1).rjust(6, '0')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': config.train_epoch + 1,
                    }, config.train_pt.format('epoch_{}'.format(epoch_)))

            log_handle.close()

            log_handle = open(config.done_log, 'w')
            timestamp = strftime("%a, %d %b %Y %H:%M:%S", localtime())
            log_handle.write(config.timestamp + '\n')
            log_handle.close()

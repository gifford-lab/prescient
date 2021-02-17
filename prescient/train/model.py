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

from .util import *

# ---- PRESCIENT

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

# shared functions and classes, including the model and `run`
# which implements the main pre-training and training loop

import torch
import torch.nn.functional as F
from torch import nn, optim

import numpy as np

from collections import OrderedDict
from types import SimpleNamespace
from time import strftime, localtime

import argparse
import itertools
import json
import os
import sys

import sklearn.decomposition

# ---- convenience functions

def p_samp(p, num_samp, w = None):
    repflag = p.shape[0] < num_samp
    p_sub = np.random.choice(p.shape[0], size = num_samp, replace = repflag)
    if w is None:
        w_ = torch.ones(len(p_sub))
    else:
        w_ = w[p_sub].clone()
    w_ = w_ / w_.sum()

    return p[p_sub,:].clone(), w_

def fit_regularizer(samples, pp, burnin, dt, sd, model, device):

    factor = samples.shape[0] / pp.shape[0]

    z = torch.randn(burnin, pp.shape[0], pp.shape[1]) * sd
    z = z.to(device)

    for i in range(burnin):
        pp = model._step(pp, dt, z = z[i,:,:])

    pos_fv = -1 * model._pot(samples).sum()
    neg_fv = factor * model._pot(pp.detach()).sum()

    return pp, pos_fv, neg_fv

def pca_transform(x):

    pca = sklearn.decomposition.PCA(n_components = 50)
    # keep track of how to break up the array after concat
    x_ = torch.cat(x)
    x_ = pca.fit_transform(x_)
    x_breaks = np.append([0], np.cumsum([len(x_) for x_ in x]))
    x_tmp = []
    for i in range(len(x_breaks) - 1):
        ii = x_breaks[i]
        jj = x_breaks[i+1]
        x_tmp.append(torch.from_numpy(x_[ii:jj]).float())
    x = x_tmp

    return x

def get_weight(w, time_elapsed):
    return w

def init(args):

    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.gpu) if args.cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    return device, kwargs

def weighted_samp(p, num_samp, w):
    ix = list(torch.utils.data.WeightedRandomSampler(w, num_samp))
    return p[ix,:].clone()

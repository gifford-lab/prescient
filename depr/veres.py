# training on veres et al. dataset

import torch
import torch.nn.functional as F 
from torch import nn, optim

import annoy
import tqdm

from geomloss import SamplesLoss

import numpy as np
import pandas as pd
import scipy.stats

from collections import OrderedDict, Counter
from joblib import Parallel, delayed 
from types import SimpleNamespace 
from time import strftime, localtime 

import argparse
import copy
import glob
import itertools
import json 
import os
import sys

import train

def init_config(args): 

    config = SimpleNamespace(

        seed = args.seed,
        timestamp = strftime("%a, %d %b %Y %H:%M:%S", localtime()), 

        # data parameters
        data_dir = args.data_dir, 
        data_path = args.data_path,
        weight_path = args.weight_path,
        weight = args.weight, 

        # model parameters 
        activation = args.activation,
        layers = args.layers, 
        k_dim = args.k_dim, 

        # pretraining parameters
        pretrain_burnin = 50, 
        pretrain_sd = 0.1, 
        pretrain_lr = 1e-9, 
        pretrain_epochs = args.pretrain_epochs, 

        # training parameters
        train_dt = args.train_dt, 
        train_sd = args.train_sd, 
        train_batch_size = args.train_batch, 
        ns = 2000,
        train_burnin = 100, 
        train_tau = args.train_tau, 
        train_epochs = args.train_epochs, 
        train_lr = args.train_lr, 
        train_clip = args.train_clip,
        save = args.save,

        # loss parameters
        sinkhorn_scaling = 0.7, 
        sinkhorn_blur = 0.1, 

        # file parameters
        out_dir = args.out_dir, 
        out_name = args.out_dir.split('/')[-1], 
        pretrain_pt = os.path.join(args.out_dir, 'pretrain.pt'), 
        train_pt = os.path.join(args.out_dir, 'train.{}.pt'), 
        train_log = os.path.join(args.out_dir, 'train.log'), 
        done_log = os.path.join(args.out_dir, 'done.log'),
        config_pt = os.path.join(args.out_dir, 'config.pt'),
    )

    config.train_t = []
    config.test_t = []
   
    if not os.path.exists(args.out_dir):
        print('Making directory at {}'.format(args.out_dir))
        os.makedirs(args.out_dir)
    else: 
        print('Directory exists at {}'.format(args.out_dir))

    return config

def load_data(config, base_dir = "."): 

    data_pt = torch.load(os.path.join(base_dir, config.data_path))
    x = data_pt['xp'] 
    y = data_pt['y']

    config.x_dim = x[0].shape[-1]
    config.t = y[-1] - y[0]
    
    y_start = y[config.start_t]
    y_ = [y_ for y_ in y if y_ > y_start]

    weight_pt = torch.load(os.path.join(base_dir, config.weight_path))

    w_ = weight_pt['w'][config.start_t]
    w = {(y_start, yy): torch.from_numpy(np.exp((yy - y_start)*w_)) for yy in y_}

    return x, y, w

def train_fate(args): 

    a = copy.copy(args)

    # data

    a.data_path = os.path.join(a.data_dir, 'fate_train.pt')

    weight = os.path.basename(a.weight_path)
    weight = weight.split('.')[0].split('-')[-1]
    a.weight = weight

    # out directory

    name = (
        "{weight}-"
        "{activation}_{layers}_{k_dim}-"
        "{train_tau}"
    ).format(**a.__dict__)

    a.out_dir = os.path.join(args.out_dir, name, 'seed_{}'.format(a.seed))
    config = init_config(a)
    
    config.start_t = 0
    config.train_t = [1, 2, 3, 4, 5, 6, 7]
    
    x, y, w = load_data(config)

    return x, y, w, config

def evaluate_fit(args, config): 
    
    log_path = os.path.join(config.out_dir, 'interpolate.log')
    if os.path.exists(log_path): 
        print(log_path, 'exists. Skipping.')
        return

    x, y, w = load_data(config) 

    # -- initialize
    device, kwargs = train.init(args) 
    model = train.AutoGenerator(config) 

    ot_solver = SamplesLoss("sinkhorn", p = 2, blur = config.sinkhorn_blur, 
        scaling = config.sinkhorn_scaling)

    losses_xy = []
    train_pts = sorted(glob.glob(config.train_pt.format('*')))
    for train_pt in train_pts: 
        
        checkpoint = torch.load(train_pt) 
        print('Loading model from {}'.format(train_pt))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(model) 

        name = os.path.basename(train_pt).split('.')[1]

        # -- evaluate
        torch.manual_seed(0)
        np.random.seed(0)

        for t_cur in config.train_t: 

            t_prev = config.start_t
            y_prev = int(y[t_prev])
            y_cur = int(y[t_cur])
        
            time_elapsed = y_cur - y_prev
            num_steps = int(np.round(time_elapsed / config.train_dt))

            dat_prev = x[t_prev].to(device)
            w_prev = train.get_weight(w[(y_prev, y_cur)], time_elapsed).cpu().numpy()

            x_s = []
            x_i_ = train.weighted_samp(dat_prev, args.evaluate_n, w_prev)

            for i in range(int(args.evaluate_n / config.ns)): 

                x_i = x_i_[i*config.ns:(i+1)*config.ns,]

                for _ in range(num_steps): 
                    z = torch.randn(x_i.shape[0], x_i.shape[1]) * config.train_sd
                    z = z.to(device)
                    x_i = model._step(x_i, dt = config.train_dt, z = z)

                x_s.append(x_i.detach())

            x_s = torch.cat(x_s)

            loss_xy = ([name, t_cur] + 
                    [ot_solver(x_s, x[t_].to(device)).item() for t_ in range(len(x))])
            losses_xy.append(loss_xy)
 
    losses_xy = pd.DataFrame(losses_xy, columns = ['epoch', 't_cur'] + y)
    losses_xy.to_csv(log_path, sep = '\t', index = False)
    print('Wrote results to', log_path) 


def main(): 
   
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type = int, default = 0)
    parser.add_argument('--no-cuda', action = 'store_true')
    parser.add_argument('--device', default = 7, type = int)
    parser.add_argument('--out_dir', default = './experiments')
    # -- data options
    parser.add_argument('--data_path')
    parser.add_argument('--data_dir')
    parser.add_argument('--weight_path', default = None)
    # -- model options
    parser.add_argument('--loss', default = 'euclidean')
    parser.add_argument('--k_dim', default = 500, type = int)
    parser.add_argument('--activation', default = 'softplus')
    parser.add_argument('--layers', default = 1, type = int)
    # -- pretrain options
    parser.add_argument('--pretrain_lr', default = 1e-9, type = float)
    parser.add_argument('--pretrain_epochs', default = 500, type = int) 
    # -- train options 
    parser.add_argument('--train_epochs', default = 5000, type = int)
    parser.add_argument('--train_lr', default = 0.01, type = float)
    parser.add_argument('--train_dt', default = 0.1, type = float)
    parser.add_argument('--train_sd', default = 0.5, type = float)
    parser.add_argument('--train_tau', default = 0, type = float)
    parser.add_argument('--train_batch', default = 0.1, type = float)
    parser.add_argument('--train_clip', default = 0.25, type = float)
    parser.add_argument('--save', default = 100, type = int)
    # -- test options
    parser.add_argument('--evaluate_n', default = 10000, type = int)
    parser.add_argument('--evaluate_data')
    parser.add_argument('--evaluate-baseline', action = 'store_true')
    # -- run options
    parser.add_argument('--task', default = 'fate')
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--evaluate')
    parser.add_argument('--config')
    args = parser.parse_args() 

    if args.task == 'fate': 

        if args.train: 

            args.pretrain = True
            args.train = True

            train.run(args, train_fate)

        if args.evaluate == 'fit': 

            if args.config:
                config = SimpleNamespace(**torch.load(args.config))
                evaluate_fit(args, config) 
            else: 
                print('Please provide a config file')


            


if __name__ == '__main__':
    main()

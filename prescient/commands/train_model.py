import os
import sys
import argparse
import copy
import numpy as np
import torch
import itertools
import json
import sklearn.decomposition

from geomloss import SamplesLoss
from collections import OrderedDict
from types import SimpleNamespace
from time import strftime, localtime

import prescient.train as train


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action = 'store_true')
    parser.add_argument('--gpu', default = 7, type = int, help="Designate GPU number as an integer (compatible with CUDA).")
    parser.add_argument('--out_dir', default = './experiments', help="Directory for storing training output.")
    parser.add_argument('--seed', type = int, default = 2, help="Set seed for training process.")
    # -- data options
    parser.add_argument('-i', '--data_path', required=True, help="Input PRESCIENT data torch file.")
    parser.add_argument('--weight_name', default = None, help="Designate descriptive name of growth parameters for filename.")
    # -- model options
    parser.add_argument('--loss', default = 'euclidean', help="Designate distance function for loss.")
    parser.add_argument('--k_dim', default = 500, type = int, help="Designate hidden units of NN.")
    parser.add_argument('--activation', default = 'softplus', help="Designate activation function for layers of NN.")
    parser.add_argument('--layers', default = 1, type = int, help="Choose number of layers for neural network parameterizing the potential function.")
    # -- pretrain options
    parser.add_argument('--pretrain_lr', default = 1e-9, type = float)
    parser.add_argument('--pretrain_epochs', default = 500, type = int, help="Number of epochs for pretraining with contrastive divergence.")
    # -- train options
    parser.add_argument('--train_epochs', default = 2500, type = int, help="Number of epochs for training.")
    parser.add_argument('--train_lr', default = 0.01, type = float, help="Learning rate for Adam optimizer.")
    parser.add_argument('--train_dt', default = 0.1, type = float, help="Timestep for simulations during training.")
    parser.add_argument('--train_sd', default = 0.5, type = float, help="Standard deviation of Gaussian noise for simulation steps.")
    parser.add_argument('--train_tau', default = 1e-6, type = float, help="Tau hyperparameter of PRESCIENT.")
    parser.add_argument('--train_batch', default = 0.1, type = float, help="Batch size for training.")
    parser.add_argument('--train_clip', default = 0.25, type = float, help="Gradient clipping threshold for training.")
    parser.add_argument('--save', default = 100, type = int, help="Save model every n epochs.")
    # -- run options
    parser.add_argument('--pretrain', type=bool, default=True, help="If True, pretraining will run.")
    parser.add_argument('--train', type=bool, default=True, help="If True, training will run with existing pretraining torch file.")
    parser.add_argument('--config')
    return parser


def init_config(args):

    config = SimpleNamespace(

        seed = args.seed,
        timestamp = strftime("%a, %d %b %Y %H:%M:%S", localtime()),

        # data parameters
        data_path = args.data_path,
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

def load_data(args):
    return torch.load(args.data_path)

def train_init(args):

    a = copy.copy(args)

    # data
    data_pt = load_data(args)
    x = data_pt["xp"]
    y = data_pt["y"]
    weight = data_pt["w"]
    if args.weight_name != None:
        a.weight = args.weight_name
    # weight = os.path.basename(a.weight_path)
    # weight = weight.split('.')[0].split('-')[-1]


    # out directory
    name = (
        "{weight}-"
        "{activation}_{layers}_{k_dim}-"
        "{train_tau}"
    ).format(**a.__dict__)

    a.out_dir = os.path.join(args.out_dir, name, 'seed_{}'.format(a.seed))
    config = init_config(a)

    config.x_dim = x[0].shape[-1]
    config.t = y[-1] - y[0]

    config.start_t = y[0]
    config.train_t = y[1:]
    y_start = y[config.start_t]
    y_ = [y_ for y_ in y if y_ > y_start]

    w_ = weight[config.start_t]
    w = {(y_start, yy): torch.from_numpy(np.exp((yy - y_start)*w_)) for yy in y_}

    return x, y, w, config

def main(args):
    train.run(args, train_init)

if __name__=="__main__":
    main()

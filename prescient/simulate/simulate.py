import warnings
warnings.filterwarnings('ignore')
import os
import sys
import argparse
import random
import joblib
import json

import numpy as np
import pandas as pd
import sklearn

from types import SimpleNamespace
from collections import Counter

def simulate(xp, y, celltype_annotations, w, model, config, num_sims, num_cells, num_steps, tp=None, celltype_subset=None, device=None):
    """
    Use trained PRESCIENT model to simulate cell trajectories with arbitrary initializations.
    """
    # load data
    xp = torch.from_numpy(xp)

    # make meta dataframe
    dict = {"tp": y, "celltype": celltype_annotations, "w": data_pt["w"]}
    meta = pd.DataFrame(dict)

    # torch parameters
    if device != None:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    all_sims = []
    for _ in range(num_sims):
        # sample cells based on timepoint or celltype or both
        if args.tp != None and args.celltype_subset != None:
            idx = pd.DataFrame(meta[(meta["tp"]==args.tp) & (meta["celltype"]==args.celltype_subset)]).sample(num_cells, weights="w")
        elif args.tp != None:
            idx = pd.DataFrame(meta[meta["tp"]==args.tp]).sample(num_cells, weights="w")
        elif args.celltype_subset != None:
            idx = pd.DataFrame(meta[meta["celltype"]==args.celltype_subset]).sample(num_cells, weights="w")
        else:
            idx = meta.sample(num_cells, weights="w")

        # map tensor to device
        xp_i = xp[idx].to(device)

        # store inital value
        xp_i_ = x_i.detach().cpu().numpy()
        xps_i = [x_i_] # n

        # simulate all cells forward through time
        for _ in range(num_steps):
            # initialize latent vector
            z = torch.randn(xp_i.shape[0], xp_i.shape[1]) * config.train_sd
            z = z.to(device)

            # step forward with trained model
            xp_i = model._step(xp_i.float(), dt=config.train_dt, z=z)

            # store next step
            xp_i_ = xp_i.detach().cpu().numpy()
            xps_i.append(x_i_)

        # group timepoints
        xps = np.stack(xps_i) #[n_cells x n_steps]
        all_sims.append(xps) #[n_sims x n_cells x n_steps]
    return all_sims

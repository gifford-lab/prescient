import warnings
warnings.filterwarnings('ignore')
import os
import sys
import argparse
import random
import joblib
import json
import tqdm
import torch

import numpy as np
import pandas as pd
import sklearn

from types import SimpleNamespace
from collections import Counter

def simulate(xp, tps, celltype_annotations, w, model, config, num_sims, num_cells, num_steps, device, tp_subset, celltype_subset):
    """
    Use trained PRESCIENT model to simulate cell trajectories with arbitrary initializations.
    """
    # load data
    xp = torch.from_numpy(xp)

    # make meta dataframe
    # TO-DO implement weight sampling strategy
    dict = {"tp": tps, "celltype": celltype_annotations}
    meta = pd.DataFrame(dict)


    all_sims = []
    pbar = tqdm.tqdm(range(num_sims))
    for s in pbar:
        # sample cells based on timepoint or celltype or both
        if tp_subset != None and celltype_subset != None:
            idx = pd.DataFrame(meta[(meta["tp"]==tp_subset) & (meta["celltype"]==celltype_subset)]).sample(num_cells).index
        elif tp_subset != None:
            idx = pd.DataFrame(meta[meta["tp"]==tp_subset]).sample(num_cells).index
        elif celltype_subset != None:
            idx = pd.DataFrame(meta[meta["celltype"]==celltype_subset]).sample(num_cells).index
        else:
            idx = meta.sample(num_cells).index

        # map tensor to device
        xp_i = xp[idx].to(device)

        # store inital value
        xp_i_ = xp_i.detach().cpu().numpy()
        xps_i = [xp_i_] # n

        # simulate all cells forward through time
        for _ in range(num_steps):
            # initialize latent vector
            z = torch.randn(xp_i.shape[0], xp_i.shape[1]) * config.train_sd
            z = z.to(device)

            # step forward with trained model
            xp_i = model._step(xp_i.float(), dt=config.train_dt, z=z)

            # store next step
            xp_i_ = xp_i.detach().cpu().numpy()
            xps_i.append(xp_i_)

        # group timepoints
        xps = np.stack(xps_i) #[n_cells x n_steps]
        all_sims.append(xps) #[n_sims x n_cells x n_steps]

        pbar.set_description('[simulate] {}'.format(s))
    return all_sims

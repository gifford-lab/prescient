import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from annoy import AnnoyIndex
import torch

def plot_interpolation(): # TO-DO
    pass

def plot_trajectory(): # TO-DO
    pass

def make_trajectory_animation(): # TO-DO
    pass

def plot_fate_streamplot(): # TO-DO
    pass

def train_ann(data_pt):
    yc = data_pt['celltype']
    xtr = data_pt["xp"]
    n_trees = 10
    n_neighbors = 20
    t = AnnoyIndex(xtr.shape[1], 'euclidean')
    for i in range(xtr.shape[0]):
        t.add_item(i, xtr[i])
    t.build(n_trees)
    return t

def classify_cells(args, data_pt, all_sims_timepoints, ann_dir):
    n_neighbors=10
    meta = data_pt["meta"]
    yc = data_pt["celltype"]
    xp_df = pd.DataFrame(data_pt["xp"], yc)
    u = AnnoyIndex(all_sims_timepoints[0][0].shape[1], 'euclidean')  # all_sims_timepoints[0][0][0].shape[1], 'euclidean')
    u.load(ann_dir)
    yp_all=[]
    for timepoint in all_sims_timepoints:
        yp=[]
        for i in range(len(timepoint)):
            yt=[]
            for j in range(len(timepoint[0])):
                nn = xp_df.iloc[u.get_nns_by_vector(timepoint[i][j], n_neighbors)]
                nn = Counter(nn.index).most_common(2)
                label, num = nn[0]
                yt.append(label)
            yp.append(yt)
        yp_all.append(yp)
    return yp_all

def compute_growth(L0, L, k, birth_score, death_score, birth_smoothed_score, death_smoothed_score):
    L = float(L)
    L0 = float(L0)
    k = float(k)

    kb = np.log(k) / np.min(birth_score)
    kd = np.log(k) / np.min(death_score)

    b = birth_smoothed_score
    d = death_smoothed_score

    b = L0 + L / (1 + np.exp(-kb * b))
    d = L0 + L / (1 + np.exp(-kd * d))
    g = b - d
    return g

def get_growth_weights(x, xp, metadata, tp_col, genes, birth_gst, death_gst, outfile,
                       n_neighbors=20, beta=0.1, L0=0.3, L=1.1, k=0.001):
    """
    Estimate growth using KEGG gene annotations. Implements smoothing procedure.

    Inputs:
    -------
    x: numpy ndarray of scaled gene expression.
    xp: numpy ndarray of PCs.
    genes: list or numpy array of highly variable gene symbols.
    birth_gst: path to csv of birth signature annotations.
    death_gst: path to csv of death signature annotations.
    outfile: provide name of outfile pt.

    Outputs:
    --------
    weights: growth rates vector.
    """
    gst = pd.read_csv(birth_gst, index_col=0)
    birth_gst = [g for g in gst['gene_symbol'].unique() if g in genes]
    gst = pd.read_csv(death_gst, index_col = 0)
    death_gst = [g for g in gst['gene_symbol'].unique() if g in genes]

    birth_gst = [g for g in birth_gst if g not in death_gst]
    death_gst = [g for g in death_gst if g not in birth_gst]

    # smoothing procedure for growth
    ay = AnnoyIndex(xp.shape[1], 'euclidean')
    for i in range(xp.shape[0]):
        ay.add_item(i, xp[i])
    ay.build(10)

    prev_score = x[birth_gst].mean(axis = 1).values
    cur_score = np.zeros(prev_score.shape)

    for _ in range(5):
        for i in range(len(prev_score)):
            xn = prev_score[ay.get_nns_by_item(i, 20)]
            cur_score[i] = (beta * xn[0]) + ((1 - beta) * xn[1:].mean(axis = 0))
        prev_score = cur_score

    birth_score = x[birth_gst].mean(axis = 1).values
    birth_smoothed_score = cur_score

    # smooth death score

    prev_score = x[death_gst].mean(axis = 1).values
    cur_score = np.zeros(prev_score.shape)

    for _ in range(5):
        for i in range(len(prev_score)):
            xn = prev_score[ay.get_nns_by_item(i, 20)]
            cur_score[i] = (beta * xn[0]) + ((1 - beta) * xn[1:].mean(axis = 0))
        prev_score = cur_score

    death_score = x[death_gst].mean(axis = 1).values
    death_smoothed_score = cur_score

    # compute growth
    g = compute_growth(L0, L, k,
                       birth_score, death_score,
                       birth_smoothed_score, death_smoothed_score)
    y_l = sorted(metadata[tp_col].unique())
    g_l = [g[(metadata[tp_col] == y_).values] for y_ in y_l]

    # write growth
    torch.save({
        "w": g_l
    }, outfile)

    return g, g_l

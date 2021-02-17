import os
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex

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
    t = annoy.AnnoyIndex(xtr.shape[1], 'euclidean')
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

def compute_growth(L0, L, k):
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

def get_growth_weights(x, xp, y, genes, gst, **kwargs):
    """
    Estimate growth using KEGG gene annotations. Implements smoothing procedure.

    Inputs:
    -------
    x: numpy ndarray of gene expression.
    genes: list or numpy array of highly variable gene symbols.
    birth_gst: birth signature annotations.

    Outputs:
    --------
    weights: growth rates vector.
    """
    birth_gst = [g for g in gst['gene_symbol'].unique() if g in genes]
    gst = pd.read_csv(gst, index_col = 0)
    death_gst = [g for g in gst['gene_symbol'].unique() if g in genes]

    birth_gst = [g for g in birth_gst if g not in death_gst]
    death_gst = [g for g in death_gst if g not in birth_gst]

    # smoothing procedure for converting to pcs
    ay = annoy.AnnoyIndex(xp_.shape[1], 'euclidean')
    for i in range(xp_.shape[0]):
        ay.add_item(i, xp_[i])
    ay.build(10)

    # compute growth
    g = compute_growth(L0, L, k)
    if args.growth_parameters != None:
        growth_args = str(args.growth_parameters).split(',')
        w = estimate_growth(x, xp, genes, args.growth_annotation, L0=growth_args[0], L=growth_args[1], k=growth_args[2])
    else:
        w = estimate_growth(x, xp, genes, y, args.growth_annotation)
    return weights

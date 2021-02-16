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

def plot_fate_streamplot() # TO-DO
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

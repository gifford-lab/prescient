

from annoy import AnnoyIndex

def train_ann(data_pt):
    yc = data_pt['celltype']
    n_trees = 10
    n_neighbors = 20
    t = annoy.AnnoyIndex(xtr.shape[1], 'euclidean')
    for i in range(xtr.shape[0]):
        t.add_item(i, xtr[i])
    t.build(n_trees)
    return t

def classify_cells(args, all_sims_timepoints, ann_dir, std):
    n_neighbors=10
    meta = std["meta"]
    yc = meta["Annotation"]
    xp_df = pd.DataFrame(std["xp_std"], yc)
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

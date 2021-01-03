import os
import numpy as np
import pandas as pd
import pyreadr
import argparse

import scanpy as sc
import anndata

import annoy
import torch

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

    return weights

def read_data(args):
    """
    - Load csv preprocessed with scanpy or Seurat.
    - Must be a csv with format n_cells x n_genes with normalized (not scaled!) expression.
    - Must have meta data with n_cells x n_metadata and include timepoints and assigned cell type labels.


    Inputs:
    -------
    path: path to csv or rds file of processed scRNA-seq dataset.
    meta: path to metadata csv.
    """
    ext = os.path.splitext(data_path)
    # load in expression dataframe
    if ext == ".txt":
        if args.meta_path == None:
            raise ValueError("Must provide path to metadata with timepoint and celltype.")
        expr = pd.read_csv(args.data_path)
        genes = expr.columns
        meta = pd.read_csv(args.meta_path)
        y = meta[args.tp_col].values.astype(int)
        celltypes = meta[args.celltype_col].values

    if ext == ".csv":
        if args.meta_path == None:
            raise ValueError("Must provide path to metadata with timepoint and ")
        expr = pd.read_csv(args.data_path)
        meta = pd.read_csv(args.meta_path)
        genes = expr.columns
        y = meta[args.tp_col].values.astype(int)
        celltypes = meta[args.celltype_col].values

    # todo: implement Scanpy anndata functionality
    if ext == ".h5ad":
        raise NotImplementedError

    # todo: implement Seurat object functionality
    if ext == ".rds":
        raise NotImplementedError

    # transformations
    scaler = sklearn.preprocessing.StandardScaler()
    pca = sklearn.decomposition.PCA(n_components = args.num_pcs)
    um = umap.UMAP(n_components = 2, metric = 'euclidean', n_neighbors = args.num_neighbors)

    if args.task == "train":
        x = scaler.fit_transform(expr)
        xp = pca.fit_transform(x)
        xu = um.fit_transform(xp)

    return x, xp, xu, y, pca, um, celltype, genes

def main():
    """
    Outputs:
    --------
    Saves a PRESCIENT file to out_path. Does not output file.
    data.pt:
        |- x: scaled expression
        |- xp: n PC space
        |- xu: UMAP space
        |- pca: sklearn pca object for pca tranformation
        |- um: umap object for umap transformation
        |- y: timepoints
        |- genes: features
        |- w: growth weights
        |- celltype: vector of celltype labels
    """
    parser = argparse.ArgumentParser()

    # file I/0
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-o', '--out_dir', type=str, required=True)
    parser.add_argument('-m', '--meta_path', type=str, required=False)

    # column names
    parser.add_argument('--tp_col', type=str, required=False)
    parser.add_argument('--celltype_col', type=str, required=False)

    # dimensionality reduction growth_parameters
    parser.add_argument('--num_pcs', type=int, default=50, required=False)
    parser.add_argument('--num_neighbors_umap', type=int, default=10, required=False)

    # proliferation scores
    parser.add_argument('--estimate_growth', type=bool, default=True)
    parser.add_argument('--growth_annotation', type=str)
    parser.add_argument('--growth_path', type=str)
    parser.add_argument('--growth_parameters', type=str, required=False)

    args = parser.parse_args()

    # throw early errors
    if "csv" in str(args.data_path).split('/')[-1] and (args.meta_path == None or args.tp_col == None or args.celltype_col == None):
        raise ValueError("If csv/tsv/txt provided, you must provide a path to metadata along with column name designations.")

    x, xp, xu, y, pca, um, celltype, genes = read_data(args)

    if not args.estimate_growth:
        w_pt = torch.load(args.growth_path)
        w = w_pt["w"]
    else:
        if args.growth_parameters not None:
            growth_args = str(args.growth_parameters).split(',')
            w = estimate_growth(x, xp, genes, args.growth_annotation L0=growth_args[0], L=growth_args[1], k=growth_args[2])
        else:
            w = estimate_growth(x, xp, genes, y, args.growth_annotation)

    # write as a torch object
    torch.save({
     "x":x,
     "xp":xp,
     "xu",xu,
     "y": y,
     "genes": genes,
     "pca": pca,
     "um":um,
     "w":w,
     "celltype": celltype
     }, args.out_dir+"data.pt")

if __name__ == '__main__':
    main()

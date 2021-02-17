import os
import numpy as np
import pandas as pd
import argparse
import pyreadr
import scanpy as sc
import anndata
import sklearn
import umap

import annoy
import torch


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
    ext = os.path.splitext(args.data_path)[1]
    # load in expression dataframe
    if ext == ".csv" or ext == ".txt" or ext == ".tsv":
        if args.meta_path == None:
            raise ValueError("Must provide path to metadata with timepoint and ")
        expr = pd.read_csv(args.data_path, index_col=0)
        meta = pd.read_csv(args.meta_path)
        genes = expr.columns
        expr = expr.to_numpy()
        tps = meta[args.tp_col].values.astype(int)
        celltype = meta[args.celltype_col].values

    # todo: implement Scanpy anndata functionality
    if ext == ".h5ad":
        raise NotImplementedError

    # todo: implement Seurat object functionality
    if ext == ".rds":
        raise NotImplementedError

    # transformations
    scaler = sklearn.preprocessing.StandardScaler()
    pca = sklearn.decomposition.PCA(n_components = args.num_pcs)
    um = umap.UMAP(n_components = 2, metric = 'euclidean', n_neighbors = args.num_neighbors_umap)

    x = scaler.fit_transform(expr)
    xp = pca.fit_transform(x)
    xu = um.fit_transform(xp)

    y = list(np.sort(np.unique(tps)))

    x_ = [torch.from_numpy(x[(meta[args.tp_col] == d),:]).float() for d in y]
    xp_ = [torch.from_numpy(xp[(meta[args.tp_col] == d),:]).float() for d in y]
    xu_ = [torch.from_numpy(xu[(meta[args.tp_col] == d),:]).float() for d in y]

    return expr, x_, xp_, xu_, y, pca, um, tps, celltype, genes

def create_parser():
    parser = argparse.ArgumentParser()

    # file I/0
    parser.add_argument('-d', '--data_path', type=str, required=True,
    help="Path to dataframe of expression values.")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
    help="Path to output directory to store final PRESCIENT data file.")
    parser.add_argument('-m', '--meta_path', type=str, required=False,
    help="Path to metadata containing timepoint and celltype annotation data.")

    # column names
    parser.add_argument('--tp_col', type=str, required=False,
    help="Column name of timepoint feature in metadate provided as string.")
    parser.add_argument('--celltype_col', type=str, required=False,
    help="Column name of celltype feature in metadata provided as string.")

    # dimensionality reduction growth_parameters
    parser.add_argument('--num_pcs', type=int, default=50, required=False,
    help="Define number of PCs to compute for input to training.")
    parser.add_argument('--num_neighbors_umap', type=int, default=10, required=False,
    help="Define number of neighbors for UMAP trasformation (UMAP used only for visualization.)")

    # proliferation scores
    parser.add_argument('--growth_path', type=str,
    help="Path to torch pt file containg pre-computed growth weights. See vignette notebooks for generating growth rate vector.")
    return parser

def main(args):
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
    # throw early errors
    if "csv" in str(args.data_path).split('/')[-1] and (args.meta_path == None or args.tp_col == None or args.celltype_col == None):
        raise ValueError("If csv/tsv/txt provided, you must provide a path to metadata along with column name designations.")

    expr, x, xp, xu, y, pca, um, tps, celltype, genes = read_data(args)

    w_pt = torch.load(args.growth_path)
    w = w_pt["w"]


    # write as a torch object
    torch.save({
     "data": expr,
     "genes": genes,
     "celltype": celltype,
     "tps": tps,
     "x":x,
     "xp":xp,
     "xu": xu,
     "y": y,
     "pca": pca,
     "um":um,
     "w":w
     }, args.out_dir+"data.pt")

if __name__ == '__main__':
    main()

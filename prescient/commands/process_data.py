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
            raise ValueError("If csv/tsv/txt provided, you must provide a path to csv metadata with timepoint and cell_type")
        if args.tp_col == None or args.celltype_col == None:
            raise ValueError("If csv/tsv/txt provided, you must provide --tp_col and --celltype_col.")

        expr = pd.read_csv(args.data_path, index_col=0)
        meta = pd.read_csv(args.meta_path)
        genes = expr.columns
        expr = expr.to_numpy()
        tps = meta[args.tp_col].values.astype(int)
        celltype = meta[args.celltype_col].values

    if ext == ".h5ad":
        adata = sc.read_h5ad(args.data_path)
        expr = adata.X
        meta = adata.obs.copy()
        try: 
            expr = expr.toarray() # In case it is in a sparse format; I do not know if this would be an obstacle downstream.
        except:
            pass
        if args.tp_col == None or args.celltype_col == None:
            raise ValueError("If h5ad input is provided, you must provide --tp_col and --celltype_col.")
        assert args.tp_col in adata.obs.columns, f"Expected a timepoint column in .obs called {args.tp_col}, but did not find it. Update the --tp_col arg?"
        assert args.celltype_col in adata.obs.columns, f"Expected a cell_type column in .obs called {args.celltype_col}, but did not find it. Update the --celltype_col arg?"
        tps = meta[args.tp_col].values.astype(int)
        celltype = meta[args.celltype_col].values
        genes = adata.var_names

    # todo: implement Seurat object functionality
    if ext == ".rds":
        raise NotImplementedError

    if args.fix_non_consecutive:
        converter = {orig:new for orig, new in zip(np.sort(np.unique(tps)), np.arange(0, len(np.unique(tps))))}
        tps = np.array([converter[orig] for orig in tps])
    assert np.all(np.sort(np.unique(tps)) == np.arange(0, len(np.unique(tps)))), "Timepoints must be labeled 0, 1, 2, ... T consecutively; no gaps are allowed"

    # transformations
    scaler = sklearn.preprocessing.StandardScaler()
    pca = sklearn.decomposition.PCA(n_components = args.num_pcs)
    um = umap.UMAP(n_components = 2, metric = 'euclidean', n_neighbors = args.num_neighbors_umap)

    x = scaler.fit_transform(expr)
    xp = pca.fit_transform(x)
    xu = um.fit_transform(xp)

    y = list(np.sort(np.unique(tps)))

    x_ = [torch.from_numpy(x[(tps == d),:]).float() for d in y]
    xp_ = [torch.from_numpy(xp[(tps == d),:]).float() for d in y]
    xu_ = [torch.from_numpy(xu[(tps == d),:]).float() for d in y]

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

    # option to fix non-consecutive timepoint labels
    parser.add_argument('--fix_non_consecutive', action="store_true", default=False,
    help="If provided, quantitative timepoints will be overwritted, e.g. 1, 4, 10 becomes 0,1,2.")

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

import os
import numpy as np
import pandas as pd
import pyreadr
import argparse

import scanpy as sc
import anndata



import annoy
import torch

def estimate_growth(x, genes, birth_gst, death_gst):
    """
    Estimate growth using KEGG gene annotations. Implements smoothing procedure.

    Inputs:
    -------
    x: numpy ndarray of gene expression.
    genes: list or numpy array of gene symbols.
    birth_gst: birth signature annotations.

    Outputs:
    --------
    weights: growth rates vector.
    """
    birth_gst = [g for g in gst['gene_symbol'].unique() if g in use_genes]
    gst = pd.read_csv(gst, index_col = 0)
    death_gst = [g for g in gst['gene_symbol'].unique() if g in use_genes]

    birth_gst = [g for g in birth_gst if g not in death_gst]
    death_gst = [g for g in death_gst if g not in birth_gst]

    # smoothing procedure for pcs
    ay = annoy.AnnoyIndex(xp_.shape[1], 'euclidean')
    for i in range(xp_.shape[0]):
        ay.add_item(i, xp_[i])
    ay.build(10)



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

    if ext == ".csv":
        if args.meta_path == None:
            raise ValueError("Must provide path to metadata with timepoint and ")
        expr = pd.read_csv(args.data_path)
        meta = pd.read_csv(args.meta_path)
        y = meta[args.tp_col].values.astype(int)

    # todo: implement Scanpy anndata functionality
    if ext == ".h5ad":
        raise NotImplementedError

    # todo: implement Seurat object functionality
    if ext == ".rds":
        raise NotImplementedError

    # transformations
    scaler = sklearn.preprocessing.StandardScaler()
    pca = sklearn.decomposition.PCA(n_components = 50)
    um = umap.UMAP(n_components = 2, metric = 'euclidean', n_neighbors = 30)

    if args.task == "train":
        x = scaler.fit_transform(expr)
        xp = pca.fit_transform(x)
        xu = um.fit_transform(xp)

    return x, xp, xu, y, pca, um, celltype, genes

    if args.task == "interpolate":
        pass

    if not args.estimate_growth:
        w_pt = torch.load(args.growth_path)
        w = w_pt["w"]




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
    parser.add_argument('-m', '--meta_path', type=str)

    # proliferation scores
    parser.add_argument('--estimate_growth', type=bool, default=True)
    parser.add_argument('--annotation_path', type=str)
    parser.add_argument('--growth_path', type=str)



    if not args.estimate_growth:
        w_pt = torch.load(args.growth_path)
        w = w_pt["w"]
    else:
        w = estimate_growth(x, xp, , args.annotation_path)

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

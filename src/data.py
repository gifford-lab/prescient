import os
import numpy as np
import pandas as pd
from scipy.io import mmread

import scanpy as sc
import anndata

import pyreadr
import torch

# def preprocess(path, meta, weights, scanpy=True, tenx=False, seurat=False):
#     """
#     Preprocess a raw data frame with scanpy.
#
#     Inputs:
#     -------
#     path: path to csv/10x raw data (if scanpy == True).
#     scanpy: use scanpy to process raw data? default=True.
#     weights: path to weights tsv.
#     seurat: if True, path is to Seurat object
#
#
#     Ouputs:
#     -------
#
#     """
#     if "csv" in path.split("/")[0]:
#         df = pd.read_csv(path)
#         meta = pd.read_csv(meta)
#         X = df.values
#         obs = meta
#         var = df.columns[1:]
#         adata = anndata.AnnData()
#     elif "h5py" in path.split("/")[0]:
#         pass
#     elif tenx:
#         # # load in 10x data format
#         # umis =  mmread(path+"/matrix.mtx")
#         # barcodes = pd.read_csv(path+"/barcodes.tsv")
#         # genes = pd.read_csv(path+"/features.tsv")
#         #
#         # # convert to array
#         # umi_array = umis.toarray()
#         #
#         # # make expression dataframe
#         # df = pd.DataFrame(data=umi_array.T, index=barcodes[0], columns=genes[1])
#         adata = sc.read_10x_mtx(path, var_names='gene_symbols', cache=True)
#
#     # follow scanpy pre-processing tutorial
#     adata.var_names_make_unique()
#     sc.pp.filter_cells(adata, min_genes=200)
#     sc.pp.filter_genes(adata, min_cells=3)
#
#     adata.var['mt'] = adata.var_names.str.startswith('mt-')
#     sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
#
#     # process raw data with scanpy
#
#
#     return df

def save_pt(path, meta_path, celltype_col, tp_col):
    """
    - Load csv preprocessed with scanpy or Seurat.
    - Must be a csv with format n_cells x n_genes with normalized (not scaled!) expression.
    - Must have meta data with n_cells x n_metadata and include timepoints and assigned cell type labels.


    Inputs:
    -------
    path: path to csv or rds file of processed scRNA-seq dataset.
    meta: path to metadata csv.
    """
    # load in expression dataframe
    expr = pd.read_csv(path)
    meta = pd.read_csv(meta_path)

    y = meta[tp_col]



    return x, xp, xu, pca, um

def save_pt(df):
    """
    Create a PRESCIENT file.
        train.pt:
        |- x: scaled expression
        |- xp: n PC space
        |- xu: UMAP space
        |- pca: sklearn pca object for pca tranformation
        |- um: umap object for umap transformation
    """
    torch.save({
    "x":x,
    "xp":xp,
    "xu",xu,
    "pca":pca,
    "um":um,
    "w":w,
    "meta": meta,
    "genes": genes
    }, out_path)
    pass

def main():
    x, xp, xu, pca,  = load_data(args.path, args.meta_path, args.celltype_col, args.tp_col)

---
title: File Formats
permalink: file_formats/
layout: document
location: file_formats
---

## File formats

PRESCIENT takes as input longitudinal scRNA-seq data. For training, all that is needed is normalized gene expression, time-point labels, and cell type annotations. These inputs are used to generate a PRESCIENT data file using `data.py`. Below, we describe accepted formatting for inputs. For pre-processing, we recommend using Seurat or scanpy. PRESCIENT accepts the following formats: **.csv**, **.tsv**, **.txt**, **.h5ad** of a scanpy anndata object, or an **.rds** file of a Seurat object.

## CSV, TSV, TXT
A post-processed gene expression file in .csv, .tsv, or .txt in the following format will work to create a PRESCIENT data object:

| id     	| gene_1 	| gene_2 	| gene_3 	| ... 	| gene_n 	|
|--------	|--------	|--------	|--------	|-----	|--------	|
| cell_1 	| 0.0    	| 0.121  	| 0.0    	|     	| 0.0    	|
| cell_2 	| 0.234  	| 0.0    	| 0.0    	|     	| 0.0    	|
| cell_3 	| 0.0    	| 0.0    	| 0.0    	|     	| 1.2    	|


## Scanpy AnnData
If pre-processing is done with Scanpy, you can directly provide the AnnData object to the PRESCIENT `data.py` command line function. The AnnData object should contain the following information:
- **adata.X** should contain a numpy ndarray, pandas DataFrame, or sparse matrix of gene expression with n_cells x n_features:

| id     	| gene_1 	| gene_2 	| gene_3 	| ... 	| gene_n 	|
|--------	|--------	|--------	|--------	|-----	|--------	|
| cell_1 	| 0.0    	| 0.121  	| 0.0    	|     	| 0.0    	|
| cell_2 	| 0.234  	| 0.0    	| 0.0    	|     	| 0.0    	|
| cell_3 	| 0.0    	| 0.0    	| 0.0    	|     	| 1.2    	|

- **adata.obs** should contain metadata of time point labels (as integers) and cell type annotations, for example:

| id     	| timepoint 	| cell_type        	|
|--------	|-----------	|------------------	|
| cell_1 	| 0         	| undifferentiated 	|
| cell_2 	| 1         	| neutrophil       	|
| cell_3 	| 2         	| monocyte         	|

## Seurat object
If pre-processing is done with Seurat, you can directly provide the Seurat object as **.rds** or convert it to a **.csv** and provide the file as directed above or provide an rds file of the Seurat object containing expressionl levels and both time-point and celltype metadata.
''

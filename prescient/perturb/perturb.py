import os
import numpy as np
import sklearn

def perturb(perturb_genes, std, pca, z_score):
    x=std["data"]
    scaler=sklearn.preprocessing.StandardScaler()
    x=scaler.fit_transform(x)
    # x_ is perturbed expression profile
    x_ = x

    genes = std["genes"]
    perturb_genes=perturb_genes.split(",")
    idx=[]

    # perturb genes that appear in highly variable gene list
    for elt in perturb_genes:
        if (elt in genes):
            idx.append(genes.index(elt))
    for elt in idx:
        x_[:,elt]= z_score
    xp = pca.transform(x_)
    return xp

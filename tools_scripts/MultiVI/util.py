import anndata
import matplotlib.pyplot as plt
import mudata as md
import muon
import scanpy as sc
import scvi
import pandas as pd
import h5py
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
import argparse
import os
import random


"""# Module 1: Data loader and preprocessing"""

def data_loader_multi_single(path, name, modality):
    adata_list = []
    for i in range(len(path)):
        current_path = path[i]
        with h5py.File(current_path, "r") as f:
            X = np.array(f['matrix/data']).transpose()
            barcodes = [bc.decode('UTF-8') for bc in f['matrix/barcodes'][:]]
            features = [feat.decode('UTF-8') for feat in f['matrix/features'][:]]
        adata = AnnData(X=X)
        adata.X = csr_matrix(np.matrix(adata.X))
        adata.var['modality'] = modality #'Gene Expression'
        features = [name + "." +  f for f in features]
        adata.var.index = features
        adata.obs['batch'] = i
        adata.obs.index = barcodes
        adata.obs.index.name = 'barcode'
        adata_list.append(adata)
    adata_list = anndata.concat(adata_list)
    return adata_list


def data_loader_multi_multi(path1, path2, name1, name2, modality1, modality2):
    adata_list = []
    for i in range(len(path1)):
        current_path1 = path1[i]
        current_path2 = path2[i]

        with h5py.File(current_path1, "r") as f1:
            with h5py.File(current_path2, "r") as f2:
                X_rna = np.mat(np.array(f1['matrix/data']).transpose())
                X_adt = np.mat(np.array(f2['matrix/data']).transpose())
                X =  (np.concatenate((X_rna,X_adt),1))
                barcodes = [bc.decode('UTF-8') for bc in f1['matrix/barcodes'][:]]
                features = [name1 + "." + key.decode('UTF-8') for key in f1['matrix/features'][:]]
                features += [name2 + "." + key.decode('UTF-8') for key in f2['matrix/features'][:]]

        adata_paired = AnnData(X=X)
        adata_paired.X =  csr_matrix(np.matrix(adata_paired.X))
        adata_paired.var['modality'] = 'all'
        adata_paired.var['modality'][0:X_rna.shape[1]] = modality1
        adata_paired.var['modality'][X_rna.shape[1]:] = modality2
        adata_paired.var.index = features

        #adata_paired.obs['batch_id'] = "1"
        adata_paired.obs['batch'] = i
        adata_paired.obs.index = barcodes
        adata_paired.obs.index.name = 'barcode'
        adata_list.append(adata_paired)
    adata_list = anndata.concat(adata_list)
    adata_list.var['modality'] = 'all'
    adata_list.var['modality'][0:X_rna.shape[1]] = modality1
    adata_list.var['modality'][X_rna.shape[1]:] = modality2
    return adata_list

def split_dataset_by_modality(adata, n, modalities):
    """
    Split an AnnData object into multiple sub-datasets based on modality.

    Parameters:
    ----------
    adata : AnnData
        The AnnData object containing the multi-modal data.
    n : int
        The number of cells in each sub-dataset.
    modalities : list of str
        The list of modalities to split the dataset by.

    Returns:
    -------
        A dictionary with modality names as keys and sub-datasets as values.
    """
    subdatasets = {}
    for i, modality in enumerate(modalities):
        subdatasets[modality] = adata[i * n : (i + 1) * n, adata.var.modality == modality].copy()
    return subdatasets

def organize_multiome_datasets(adata_modality1, adata_modality2, adata_paired):
    """
    Organizes multiple sub-datasets into a single Multiome dataset.

    Parameters:
    ----------
    adata_modality1 : AnnData
        The AnnData object containing the first modality data.
    adata_modality2 : AnnData
        The AnnData object containing the second modality data.
    adata_paired : AnnData
        The AnnData object containing the paired modality data

    Returns:
    -------
    AnnData
        The organized Multiome dataset.
    """
    return scvi.data.organize_multiome_anndatas(adata_modality1, adata_modality2, adata_paired)

def sort_features_by_modality(adata):
    """
    Sorts the features of an AnnData object by modality, ensuring that expression data appear before accessibility data.

    Parameters:
    ----------
    adata : AnnData
        The AnnData object containing the multi-modal data.

    Returns:
    -------
    AnnData
        The AnnData object with features sorted by modality.
    """
    return adata[:, adata.var["modality"].argsort()].copy()

def filter_features(adata, min_cells_pct):
    """
    Filters features that appear in fewer than the specified percentage of cells.

    Parameters:
    ----------
    adata : AnnData
        The AnnData object containing the data to be filtered.
    min_cells_pct : float
        The minimum percentage of cells in which a feature must appear to be kept.

    Returns:
    -------
    AnnData
        The filtered AnnData object.
    """
    min_cells = int(adata.shape[0] * min_cells_pct / 100)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata

def setup_anndata_for_multivi(adata, batch_key):
    """
    Sets up the AnnData object for training the MultiVI model.
    Parameters:
    ----------
    adata : AnnData
        The AnnData object containing the multi-modal data.
    batch_key : str
        The key in the AnnData object's .obs attribute that corresponds to the modality of the cells.
    Returns:
    -------
    None
    """
    scvi.model.MULTIVI.setup_anndata(adata, batch_key=batch_key)

def create_multivi_model(adata):
    """
    Creates a MultiVI model object.
    Parameters:
    ----------
    adata : AnnData
        The AnnData object containing the multi-modal data.
    Returns:
    -------
    scvi.model.MULTIVI
        The MultiVI model object.
    """
    n_rna = (adata.var["modality"] == "Gene Expression").sum()
    n_atac = (adata.var["modality"] == "Peak").sum()
    print(f"Number of RNA features: {n_rna}")
    print(f"Number of ATAC features: {n_atac}")
    return scvi.model.MULTIVI(adata, n_genes=n_rna, n_regions=n_atac)

"""# Module2 Creating, saving and loading the model"""

def train_multivi_model(model, adata, n_epochs=400, lr=1e-3, use_cuda=True):
    accelerator = "gpu" if use_cuda else "cpu"
    model.train(
        max_epochs=n_epochs,
        lr=lr,
        #accelerator=accelerator
    )

def save_multivi_model(model, model_dir, overwrite=True):
    model.save(model_dir, overwrite=overwrite)

def load_multivi_model(model_dir, adata):
    return scvi.model.MULTIVI.load(model_dir, adata=adata)

"""# Module3: Functions"""

def get_normalized_expression(model, adata=None):
    """
    Returns the expected value of gene expression under the approximate posterior.

    Parameters:
    ----------
    model : MultiVI
        The MultiVI model.
    n_samples : int, optional
        The number of samples to use for Monte Carlo approximation.

    Returns:
    -------
    pd.DataFrame
        The expected value of gene expression.
    """
    return model.get_normalized_expression(adata)

def get_accessibility_estimates(model, adata = None):
    """
    Returns the expected value of accessibility under the approximate posterior.

    Parameters:
    ----------
    model : MultiVI
        The MultiVI model.
    n_samples_overall : int, optional
        The number of samples to use for Monte Carlo approximation.

    Returns:
    -------
    pd.DataFrame
        The expected value of accessibility.
    """
    return model.get_accessibility_estimates(adata)

"""# Module4: Combination of the functions"""

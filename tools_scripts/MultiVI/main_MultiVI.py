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
from util import data_loader_multi_single, data_loader_multi_multi, split_dataset_by_modality, organize_multiome_datasets, sort_features_by_modality, filter_features, setup_anndata_for_multivi, get_normalized_expression, create_multivi_model, train_multivi_model, get_accessibility_estimates, save_multivi_model, load_multivi_model

random.seed(1)
parser = argparse.ArgumentParser("MultiVI")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train data1')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train data2')
parser.add_argument('--pair_path1', metavar='DIR', default='NULL', help='path to train pair data1')
parser.add_argument('--pair_path2', metavar='DIR', default='NULL', help='path to train pair data2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

"""# Module2 Creating, saving and loading the model"""


def run_MultiVI(file_paths, n_epochs=500, lr=1e-3):

    # Load and preprocess the data
    rna_path = file_paths["rna_path_single"]
    atac_path = file_paths["atac_path_single"]
    adata_single_rna = data_loader_multi_single(rna_path, 'rna', "Gene Expression")
    adata_single_atac = data_loader_multi_single(atac_path, 'atac', "Peak")
    adata_multi = data_loader_multi_multi(file_paths['rna_path_paired'], file_paths['atac_path_paired'], "rna","atac", "Gene Expression", "Peak")
    # Organize the multiome datasets
    adata = organize_multiome_datasets(adata_multi,adata_single_rna, adata_single_atac)
    # Sort features by modality
    adata = sort_features_by_modality(adata)
    # Filter features
    adata = filter_features(adata,1)
    # Get the names of all features (genes and peaks) after filtering
    genes_to_impute = adata.var.index.tolist()
    # Setup anndata for multivi
    setup_anndata_for_multivi(adata, batch_key="modality")
    # Create the model
    model = create_multivi_model(adata)
    train_multivi_model(model, adata, n_epochs=n_epochs, lr=lr)
    normalized_expression = get_normalized_expression(model)
    accessibility_estimates = get_accessibility_estimates(model)
    return model, adata, adata_single_rna, adata_single_atac, adata_multi, normalized_expression, accessibility_estimates

file_paths = {
    "rna_path_single": [
        args.path1
    ],
    "atac_path_single": [
        args.path2
    ],
    "rna_path_paired": [
        args.pair_path1
    ],
    "atac_path_paired": [
        args.pair_path2
    ]
}
n_epochs = 500
lr = 1e-3
model, adata, adata_single_rna, adata_single_atac, adata_multi, normalized_expression, accessibility_estimates = run_MultiVI(
    file_paths,n_epochs
)

latent_key = "X_multivi"
adata.obsm[latent_key] = model.get_latent_representation()
result = adata.obsm[latent_key]
result1 = result[0:adata_single_atac.shape[0],:]
result1 = np.transpose(result1)
result2 = result[adata_single_atac.shape[0]:(adata_single_atac.shape[0]+adata_single_rna.shape[0]),:]
result2 = np.transpose(result2)
result3 = result[(adata_single_atac.shape[0]+adata_single_rna.shape[0]):,:]
result3 = np.transpose(result3)
result = np.concatenate([result2, result3, result1], 1)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()

file = h5py.File(args.save_path+"/imputed_rna.h5", 'w')
file.create_dataset('data', data=np.transpose(normalized_expression))
file.close()

file = h5py.File(args.save_path+"/imputed_atac.h5", 'w')
file.create_dataset('data', data=np.transpose(accessibility_estimates))
file.close()

import os
import time
import h5py
import muon
import random
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import multigrate as mtg
import scipy.sparse as sp
from scipy.sparse import csr_matrix

random.seed(1)
parser = argparse.ArgumentParser("Multigrate")
parser.add_argument('--path1', metavar='DIR', default="", help='path to train data1')
parser.add_argument('--path2', metavar='DIR', default="", help='path to train data2')
parser.add_argument('--path3', metavar='DIR', default="", help='path to train data2')
parser.add_argument('--path4', metavar='DIR', default="", help='path to train data2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
parser.add_argument('--epochs', type = int, default=200, help='')
parser.add_argument('--lr', type = float, default=1e-3, help='')

args = parser.parse_args()

begin_time = time.time()


def process_rna(adata_rna,barcodes,featurs):
    adata_rna.layers['counts'] = adata_rna.X
    adata_rna.obs.index = barcodes
    adata_rna.var_names = featurs
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    return adata_rna
    
def process_atac(adata_atac,barcodes,featurs):
    adata_atac.layers['counts'] = adata_atac.X
    adata_atac.obs.index = barcodes
    adata_atac.var_names = featurs
    sc.pp.normalize_total(adata_atac, target_sum=1e4)
    sc.pp.log1p(adata_atac)
    adata_atac.layers['log-norm'] = adata_atac.X.copy()
    return adata_atac

def process_atac(adata_atac,barcodes,featurs):
    adata_atac.layers['counts'] = adata_atac.X
    adata_atac.obs.index = barcodes
    adata_atac.var_names = featurs
    muon.prot.pp.clr(adata_atac)
    return adata_atac
    

def h5_to_matrix(path):
    with h5py.File(path, "r") as f:
        X = csr_matrix(np.mat(np.array(f['matrix/data']).transpose()))
        barcodes = [key.decode('UTF-8') for key in f['matrix/barcodes']]
        features = [key.decode('UTF-8') for key in f['matrix/features']]
    return X, barcodes, features
    
    
rna_path1 = args.path1
rna_path2 = args.path2
atac_path2 = args.path3
atac_path3 = args.path4

rna1, barcodes1, rna_features = h5_to_matrix(rna_path1)
rna2, barcodes2, rna_features = h5_to_matrix(rna_path2)
atac2, barcodes2, atac_features = h5_to_matrix(atac_path2)
atac3, barcodes3, atac_features = h5_to_matrix(atac_path3)


adata_rna1 = process_rna(ad.AnnData(rna1),barcodes1, rna_features)
adata_rna2 = process_rna(ad.AnnData(rna2),barcodes2, rna_features)
adata_atac2 = process_atac(ad.AnnData(atac2), barcodes2, atac_features)
adata_atac3 = process_atac(ad.AnnData(atac3), barcodes3, atac_features)

adata_rna1.obs['Modality'] = "rna"
adata_rna2.obs['Modality'] = "cite"
adata_atac2.obs['Modality'] = "cite"
adata_atac3.obs['Modality'] = "atac"

adata_rna = ad.concat([adata_rna1, adata_rna2], merge='same')
sc.pp.highly_variable_genes(adata_rna, n_top_genes=4000,  batch_key='Modality')
adata_rna1 = adata_rna1[:, adata_rna.var.highly_variable]
adata_rna2 = adata_rna2[:, adata_rna.var.highly_variable]

adata_atac = ad.concat([adata_atac2, adata_atac3], merge='same')
sc.pp.highly_variable_genes(adata_atac, n_top_genes=20000,  batch_key='Modality')
adata_atac2 = adata_atac2[:, adata_atac.var.highly_variable]
adata_atac3 = adata_atac3[:, adata_atac.var.highly_variable]

print(adata_rna2.obs.index == adata_atac2.obs.index)
indices = range(1, len(adata_rna1.obs_names) + 1)
adata_rna1.obs.index= [f"{index}rna-{original}" for index, original in zip(indices, adata_rna1.obs_names)]
indices = range(1, len(adata_rna2.obs_names) + 1)
adata_rna2.obs.index= [f"{index}cite-{original}" for index, original in zip(indices, adata_rna2.obs_names)]
indices = range(1, len(adata_atac2.obs_names) + 1)
adata_atac2.obs.index= [f"{index}cite-{original}" for index, original in zip(indices, adata_atac2.obs_names)]
indices = range(1, len(adata_atac3.obs_names) + 1)
adata_atac3.obs.index= [f"{index}atac-{original}" for index, original in zip(indices, adata_atac3.obs_names)]

adata = mtg.data.organize_multiome_anndatas(
    adatas = [[adata_rna1, adata_rna2, None], [None, adata_atac2, adata_atac3]],
    layers = [['counts', 'counts', None], [None, 'log-norm', 'log-norm']],
)
adata


    
mtg.model.MultiVAE.setup_anndata(
    adata,
    categorical_covariate_keys=['Modality'],# 'Samplename'],
    rna_indices_end=4000
)

model = mtg.model.MultiVAE(
    adata,
    losses=['nb', 'mse'],
    loss_coefs={'kl': 1e-3,
               'integ': 4000,
               },
    integrate_on='Modality',
    mmd='marginal',
)

model.train()

model.get_latent_representation()
result = adata.obsm['latent']


end_time = time.time()
all_time = end_time - begin_time
print(all_time)
print(result.shape)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")


file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()

np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")

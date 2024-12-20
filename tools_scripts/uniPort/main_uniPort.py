import os
import h5py
import time
import torch
import random
import scipy.io
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import uniport as up
from anndata import AnnData
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser("uniPort")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to RNA')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to ATAC')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
parser.add_argument('--seed',  type=int,  default=1, help='path to save the output data')
args = parser.parse_args()

# The uniPort script for diagonal integration requires RNA and ATAC data as input, where ATAC needs to be transformed into gene activity score. The output is a joint embedding (dimensionality reduction).
# run commond for uniPort
# python main_uniPort.py --path1 "../../data/dataset_final/D27/rna.h5" --path2 "../../data/dataset_final/D27/atac_gas.h5" --save_path "../../result/embedding/diagonal integration/D27/uniPort/"

begin_time = time.time()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.mat(np.array(f['matrix/data']).transpose())
    adata = AnnData(X=csr_matrix(X))
    return adata
    
adata_rna = data_loader(args.path1)
adata_atac = data_loader(args.path2)

adata_rna.obs['domain_id'] = 1
adata_rna.obs['domain_id'] = adata_rna.obs['domain_id'].astype('category')
adata_rna.obs['source'] = 'RNA'

adata_atac.obs['domain_id'] = 0
adata_atac.obs['domain_id'] = adata_atac.obs['domain_id'].astype('category')
adata_atac.obs['source'] = 'ATAC'

adata_cm = adata_atac.concatenate(adata_rna, join='inner', batch_key='domain_id')

sc.pp.normalize_total(adata_cm)
sc.pp.log1p(adata_cm)
sc.pp.highly_variable_genes(adata_cm, n_top_genes=2000, inplace=False, subset=True)
up.batch_scale(adata_cm)
print(adata_cm.obs)

sc.pp.normalize_total(adata_rna)
sc.pp.log1p(adata_rna)
sc.pp.highly_variable_genes(adata_rna, n_top_genes=2000, inplace=False, subset=True)
up.batch_scale(adata_rna)

sc.pp.normalize_total(adata_atac)
sc.pp.log1p(adata_atac)
sc.pp.highly_variable_genes(adata_atac, n_top_genes=2000, inplace=False, subset=True)
up.batch_scale(adata_atac)

adata = up.Run(adatas=[adata_atac,adata_rna], adata_cm=adata_cm, lambda_s=1.0, seed=args.seed)

embedding = adata.obsm['latent']
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
print(embedding.shape)
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=(embedding))
file.close()








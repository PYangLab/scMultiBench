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
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train data1')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train data2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
parser.add_argument('--seed',  type=int,  default=1, help='path to save the output data')
parser.add_argument('--cty_path1', metavar='DIR',default='NULL', help='path to cty1')
parser.add_argument('--cty_path2', metavar='DIR', default='NULL', help='path to cty2')
args = parser.parse_args()
begin_time = time.time()

# this script is for classification
# run commond for uniPort
# python main_uniPort_classification.py --path1 "../../data/dataset_final/D27/rna.h5" --path2 "../../data/dataset_final/D27/atac_gas.h5" --cty_path1 "../../data/dataset_final/D27/rna_cty.csv" --cty_path2 "../../data/dataset_final/D27/atac_cty.csv" --save_path "../../result/classification/D27/uniPort_ori/"

def read_fs_label(label_path):
    label = pd.read_csv(label_path,header=None,index_col=False)  #
    label = label.iloc[1:(label.shape[0]),0]
    print(label)
    return label
    
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
rna_labels = read_fs_label(args.cty_path1)
atac_labels = read_fs_label(args.cty_path2)

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
rna_labels.index = adata_rna.obs.index
adata_rna.obs['celltype'] = rna_labels.astype('category')

adata_atac.obs['domain_id'] = 0
adata_atac.obs['domain_id'] = adata_atac.obs['domain_id'].astype('category')
adata_atac.obs['source'] = 'ATAC'
atac_labels.index = adata_atac.obs.index
adata_atac.obs['celltype'] = atac_labels.astype('category')

print("ori rna", adata_rna)
print("ori atac", adata_atac)

common_celltypes = set(adata_rna.obs['celltype']).intersection(set(adata_atac.obs['celltype']))

adata_rna = adata_rna[adata_rna.obs['celltype'].isin(common_celltypes)]
adata_atac = adata_atac[adata_atac.obs['celltype'].isin(common_celltypes)]
print(adata_rna.obs)
print("filted rna", adata_rna)
print("filted atac", adata_atac)


adata_cm = adata_atac.concatenate(adata_rna, join='inner', batch_key='domain_id')

# sc.pp.highly_variable_genes(adata_cm, n_top_genes=2000, flavor="seurat_v3")
sc.pp.normalize_total(adata_cm)
sc.pp.log1p(adata_cm)
sc.pp.highly_variable_genes(adata_cm, n_top_genes=2000, inplace=False, subset=True)
up.batch_scale(adata_cm)
# sc.pp.scale(adata_cm)
print(adata_cm.obs)

# sc.pp.highly_variable_genes(adata_rna, n_top_genes=2000, flavor="seurat_v3")
sc.pp.normalize_total(adata_rna)
sc.pp.log1p(adata_rna)
sc.pp.highly_variable_genes(adata_rna, n_top_genes=2000, inplace=False, subset=True)
up.batch_scale(adata_rna)
# sc.pp.scale(adata_rna)

# sc.pp.highly_variable_genes(adata_atac, n_top_genes=2000, flavor="seurat_v3")
sc.pp.normalize_total(adata_atac)
sc.pp.log1p(adata_atac)
sc.pp.highly_variable_genes(adata_atac, n_top_genes=2000, inplace=False, subset=True)
up.batch_scale(adata_atac)
# sc.pp.scale(adata_atac)

adata = up.Run(adatas=[adata_atac,adata_rna], adata_cm=adata_cm, lambda_s=1.0, seed=args.seed)

embedding = adata.obsm['latent']

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")


sc.pp.neighbors(adata, use_rep='latent')
sc.tl.umap(adata, min_dist=0.1)
sc.pl.umap(adata, color=['source', 'celltype'], wspace=0.3, legend_fontsize=10)

adata1 = adata[adata.obs['domain_id']=='0']
adata2 = adata[adata.obs['domain_id']=='1']
y_test = up.metrics.label_transfer(adata2, adata1, label='celltype', rep='X_umap')
print(sum(y_test==adata1.obs["celltype"])/len(y_test))
pd.DataFrame(y_test).to_csv(os.path.join(args.save_path, "predict.csv"))
pd.DataFrame(adata1.obs["celltype"]).to_csv(os.path.join(args.save_path, "query.csv"))







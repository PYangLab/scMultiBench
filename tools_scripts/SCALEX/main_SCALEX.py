import os
import h5py
import time
import random
import argparse
import numpy as np
import scanpy as sc
from anndata import AnnData
from scalex import SCALEX

random.seed(1)
###
parser = argparse.ArgumentParser("SCALEX")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train data1')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train data2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()
begin_time = time.time()

def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
    adata = AnnData(X=(X))
    return adata

adata_rna = data_loader(args.path1)
adata_atac = data_loader(args.path2)
    
adata = adata_rna.concatenate(adata_atac, join='inner', batch_key='batch')
adata.write('RNA-ATAC.h5ad')


sc.pp.filter_genes(adata, min_cells=0)
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
sc.tl.umap(adata, min_dist=0.1)

wk_dir='./'
adata = SCALEX(data_list = [wk_dir+'RNA-ATAC.h5ad'],
              min_features=0,
              min_cells=0,
              outdir=wk_dir+'/pbmc_RNA_ATAC/',
              show=False,
              gpu=0)
              
            
embedding = adata.obsm['latent']

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
    
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=embedding)
file.close()

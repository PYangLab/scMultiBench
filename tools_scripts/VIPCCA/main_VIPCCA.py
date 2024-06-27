import os
import h5py
import random
import argparse
import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix
import vipcca.tools.utils as tl
import vipcca.model.vipcca as vip
import vipcca.tools.plotting as pl
import vipcca.tools.transferLabel as tfl

random.seed(42)
parser = argparse.ArgumentParser("VIPCCA")
# Data configs
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train gene')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train peak')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.mat(np.array(f['matrix/data']).transpose())
    adata = AnnData(X=csr_matrix(X))
    return adata

adata_rna = data_loader(args.path1)
adata_atac = data_loader(args.path2)

adata_rna.obs['tech'] = ['rna' for _ in range(adata_rna.n_obs)]
adata_rna.obs['_batch'] = ['RNA' for _ in range(adata_rna.n_obs)]
adata_atac.obs['tech'] = ['atac' for _ in range(adata_atac.n_obs)]
adata_atac.obs['_batch'] = ['ATAC' for _ in range(adata_atac.n_obs)]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
adata_all= tl.preprocessing([adata_rna, adata_atac])
handle = vip.VIPCCA(adata_all=adata_all,
                           res_path='./atac_result/',
                           mode='CVAE',
                           split_by="_batch",
                           epochs=20,
                           lambda_regulizer=2,
                           batch_input_size=64,
                           batch_input_size2=14,
                           )
adata_integrate=handle.fit_integrate()
result = adata_integrate.obsm['X_vipcca']

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")

file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()


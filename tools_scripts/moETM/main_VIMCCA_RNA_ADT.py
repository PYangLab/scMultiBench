import os
import h5py
import random
import argparse
import numpy as np
import scanpy as sc
from anndata import AnnData
from scbean.model import vimcca
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser("VIMCCA")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to RNA')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to ADT')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

# The VIMCCA script for vertical integration requires RNA and ADT data as input. The output is a joint embedding (dimensionality reduction).
# run commond for VIMCCA
# python main_VIMCCA_RNA_ADT.py --path1 "../../data/dataset_final/D3/rna.h5" --path2 "../../data/dataset_final/D3/adt.h5" --save_path "../../result/embedding/diagonal integration/D3/VIMCCA/"

def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
    adata = AnnData(X=csr_matrix(X))
    return adata

adata_rna = data_loader(args.path1)
adata_adt = data_loader(args.path2)

sc.pp.filter_genes(adata_rna, min_cells=10)
sc.pp.log1p(adata_rna)
sc.pp.log1p(adata_adt)
sc.pp.scale(adata_rna)
sc.pp.scale(adata_adt)
result = vimcca.fit_integration(
    adata_rna,
    adata_adt,
    sparse_x=False,
    sparse_y=False,
    hidden_layers=[128, 64, 32, 8],
    epochs=50
)
print(result.shape)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()


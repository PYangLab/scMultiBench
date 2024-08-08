import os
import time
import scvi
import h5py
import anndata
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser("totalVI")
parser.add_argument('--path1', metavar='DIR', nargs='+', default=[], help='path to train data1')
parser.add_argument('--path2', metavar='DIR', nargs='+', default=[], help='path to train data2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()
begin_time = time.time()

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
    adata = AnnData(X=X)
    return adata
    
    
def run_totalVI(file_paths):
    rna_path=file_paths['rna_path']
    adt_path=file_paths['adt_path']
    rna_list = []
    adt_list = []
    
    # read rna
    if rna_path is not None:
        for i in range(len(rna_path)):
            if rna_path[i] is None:
                rna_list.append(None)
            else:
                rna_list.append(data_loader(rna_path[i]))
    batch = []
    for i in range(len(rna_list)):
        batch.append(np.ones((rna_list[i].shape[0], 1))  + i)
    batch = np.concatenate(batch,0)
    adata = anndata.concat(rna_list)
    adata.obs["batch"] = batch
    
    # read adt
    if adt_path is not None:
        for i in range(len(adt_path)):
            if adt_path[i] is None:
                adt_list.append(None)
            else:
                adt_list.append(data_loader(adt_path[i]))
    adata_adt = anndata.concat(adt_list)
    adt = (adata_adt.X)
    a = [i for i in range(0,adata.shape[0])]
    (adata.obs).index = a
    adata.layers["counts"] = adata.X.copy()
    adata.obsm['protein_expression'] = adt
    print(np.max(adt))
    
    
    scvi.model.TOTALVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        layer="counts",
        batch_key="batch"
    )
    vae = scvi.model.TOTALVI(adata, empirical_protein_background_prior=False,latent_distribution="normal")
    vae.train()
    
    result_embedding = vae.get_latent_representation()
 
    return result_embedding

file_paths = {
    "rna_path": args.path1,
    "adt_path": args.path2
}

result = run_totalVI(file_paths)
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

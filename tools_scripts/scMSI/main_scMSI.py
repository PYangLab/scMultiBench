
import os
import sys
import time
import h5py
import torch
import random
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy.linalg import norm
from scMSI.utils import read_txt, init_library_size
from scMSI.utils import get_top_coeff, get_knn_Aff
from scMSI.scMSI_main import SCMSIRNA, SCMSIProtein, SCMSIRNAProtein, SCMSICiteRna

parser = argparse.ArgumentParser("scMSI")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train rna')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train adt')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

# The scMSI script for vertical integration requires RNA and ADT data as input. The output is a joint embedding (dimensionality reduction).
# run commond for scMSI
# vertical integration:
# python main_scMSI.py --path1 "../../data/dataset_final/D3/rna.h5" --path2 "../../data/dataset_final/D3/adt.h5"  --save_path "../../result/embedding/D3/"

begin_time = time.time()
def run_scMSI(adt_path,rna_path):
    with h5py.File(adt_path, 'r') as f:
        data_adt = np.array(f['matrix/data'])
        barcodes_adt = np.array(f['matrix/barcodes'])
        features_adt = np.array(f['matrix/features'])
    with h5py.File(rna_path, 'r') as f:
        data_rna = np.array(f['matrix/data'])
        barcodes_rna = np.array(f['matrix/barcodes'])
        features_rna = np.array(f['matrix/features'])

    RNA_data = sc.AnnData(X=data_rna.T, obs=pd.DataFrame(index=barcodes_rna), var=pd.DataFrame(index=features_rna))
    ADT_data = sc.AnnData(X=data_adt.T, obs=pd.DataFrame(index=barcodes_adt), var=pd.DataFrame(index=features_adt))
    gene_name = []
    for name in RNA_data.var_names:
      # Convert bytes to string if necessary
      if isinstance(name, bytes):
          name = name.decode("utf-8")

      # Directly append the gene name without splitting
      gene_name.append(name)
    RNA_data.var_names = gene_name
    RNA_data.uns["protein_names"] = np.array(ADT_data.var_names)
    RNA_data.obsm["protein_expression"] = ADT_data.X
    RNA_data.layers["rna_expression"] = RNA_data.X.copy()
    adata = RNA_data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    scMSI_output_times10 = pd.DataFrame()
    model = SCMSIRNAProtein(
      adata,
      n_latent=10,
      latent_distribution="normal",
    )
    max_epochs = 400
    model.train(max_epochs=max_epochs, record_loss=True)
    rna_latent, pro_latent = model.get_latent_representation(batch_size=128)
    adata.obsm["rna_latent"] = rna_latent
    adata.obsm["pro_latent"] = pro_latent
    mix_latent = np.concatenate([adata.obsm["rna_latent"], adata.obsm["pro_latent"]], axis=1)
    return mix_latent

# RUN METHOD
result = run_scMSI(args.path2,args.path1)
end_time = time.time()
all_time = end_time - begin_time
result = np.transpose(result)
print(result)
# SAVE RESULT
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")
# SAVE RESULT

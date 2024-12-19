import os
import h5py
import time
import portal
import anndata
import scipy.io
import argparse
import numpy as np
import pandas as pd
import scanpy as sc

parser = argparse.ArgumentParser("Portal")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to rna')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to atac')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

# The Portal script for diagonal integration requires RNA and ATAC data as input, where ATAC needs to be transformed into gene activity score. The output is a joint embedding (dimensionality reduction).
# run commond for Portal
# python main_Portal.py --path1 "../../data/dataset_final/D27/rna.h5" --path2 "../../data/dataset_final/D27/atac_gas.h5" --save_path "../../result/embedding/diagonal integration/D27/Portal/"

begin_time = time.time()
rna_h5_files = args.path1
h5 = h5py.File(rna_h5_files, "r")
h5_rna_data = h5['matrix/data']
h5_rna_barcodes = h5['matrix/barcodes']
h5_rna_features = h5['matrix/features']
rna_data = scipy.sparse.csr_matrix(np.array(h5_rna_data).transpose()).copy()
rna_barcodes = np.array(h5_rna_barcodes).astype(str).copy()
rna_features = np.array(h5_rna_features).astype(str).copy()
adata_rna = anndata.AnnData(rna_data)
adata_rna.obs.index = rna_barcodes
adata_rna.obs["data_type"] = "rna"
adata_rna.var.index = rna_features

atac_h5_files = args.path2
h5 = h5py.File(atac_h5_files, "r")
h5_atac_data = h5['matrix/data']
h5_atac_barcodes = h5['matrix/barcodes']
h5_atac_features = h5['matrix/features']
atac_data = scipy.sparse.csr_matrix(np.array(h5_atac_data).transpose()).copy()
atac_barcodes = np.array(h5_atac_barcodes).astype(str).copy()
atac_features = np.array(h5_atac_features).astype(str).copy()
adata_atac = anndata.AnnData(atac_data)
adata_atac.obs.index = atac_barcodes
adata_atac.obs["data_type"] = "atac"
adata_atac.var.index = atac_features

meta_rna = adata_rna.obs
meta_atac = adata_atac.obs
meta = pd.concat([meta_rna, meta_atac], axis=0)

model = portal.model.Model(training_steps=3000, lambdacos=10.0)
model.preprocess(adata_rna, adata_atac) # perform preprocess and PCA
model.train() # train the model
model.eval() # get integrated latent representation of cells
result = model.latent

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")

file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=np.transpose(result))
file.close()


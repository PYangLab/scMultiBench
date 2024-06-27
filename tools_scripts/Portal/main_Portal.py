import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import csv
import gzip
import scipy.io
import portal
import argparse
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(1)

sc.settings.verbosity = 3
sc.logging.print_header()

###
parser = argparse.ArgumentParser("Portal")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train data1')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train data2')
parser.add_argument('--cty_path1', metavar='DIR', default='NULL', help='path to train cty1')
parser.add_argument('--cty_path2', metavar='DIR', default='NULL', help='path to train cty2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()
begin_time = time.time()

rna_h5_files = args.path1
rna_label_files = args.cty_path1
h5 = h5py.File(rna_h5_files, "r")
h5_rna_data = h5['matrix/data']
h5_rna_barcodes = h5['matrix/barcodes']
h5_rna_features = h5['matrix/features']
rna_data = scipy.sparse.csr_matrix(np.array(h5_rna_data).transpose()).copy()
rna_barcodes = np.array(h5_rna_barcodes).astype(str).copy()
rna_features = np.array(h5_rna_features).astype(str).copy()
rna_label = pd.read_csv(rna_label_files, index_col = 0)
adata_rna = anndata.AnnData(rna_data)
adata_rna.obs.index = rna_barcodes
adata_rna.obs["cell_type"] = rna_label["x"].values.astype(str)
adata_rna.obs["data_type"] = "rna"
adata_rna.var.index = rna_features

atac_h5_files = args.path2
atac_label_files = args.cty_path2
h5 = h5py.File(atac_h5_files, "r")
h5_atac_data = h5['matrix/data']
h5_atac_barcodes = h5['matrix/barcodes']
h5_atac_features = h5['matrix/features']
atac_data = scipy.sparse.csr_matrix(np.array(h5_atac_data).transpose()).copy()
atac_barcodes = np.array(h5_atac_barcodes).astype(str).copy()
atac_features = np.array(h5_atac_features).astype(str).copy()
atac_label = pd.read_csv(atac_label_files, index_col = 0)
adata_atac = anndata.AnnData(atac_data)
adata_atac.obs.index = atac_barcodes
adata_atac.obs["cell_type"] = atac_label["x"].values.astype(str)
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


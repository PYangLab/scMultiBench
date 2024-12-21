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
import torch
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sc.settings.verbosity = 3
sc.logging.print_header()

parser = argparse.ArgumentParser("Portal")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train data1')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train data2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
parser.add_argument('--cty_path1', metavar='DIR',default='NULL', help='path to cty1')
parser.add_argument('--cty_path2', metavar='DIR', default='NULL', help='path to cty2')
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()
begin_time = time.time()

# The Portal script for classification.
# run commond for Portal
# python main_Portal.py --path1 "../../data/dataset_final/D27/rna.h5" --path2 "../../data/dataset_final/D27/atac_gas.h5" --save_path "../../result/embedding/diagonal integration/D27/Portal/"

torch.manual_seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def read_fs_label(label_path):
    label = pd.read_csv(label_path,header=None,index_col=False)  #
    label = label.iloc[1:(label.shape[0]),0]
    print(label)
    return label
rna_labels = read_fs_label(args.cty_path1)
atac_labels = read_fs_label(args.cty_path2)

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
rna_labels.index = adata_rna.obs.index
adata_rna.obs['celltype'] = rna_labels.astype('category')

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


meta_rna = adata_rna.obs
meta_atac = adata_atac.obs
meta = pd.concat([meta_rna, meta_atac], axis=0)
print(meta, "!!!!!!")
model = portal.model.Model(training_steps=3000, lambdacos=10.0, seed=args.seed)
model.preprocess(adata_rna, adata_atac) # perform preprocess and PCA
model.train() # train the model
model.eval() # get integrated latent representation of cells
result = model.latent


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")

from scipy.spatial import distance_matrix
def annotate_by_nn(vec_tar, vec_ref, label_ref, k=20):
    dist_mtx = distance_matrix(vec_tar, vec_ref)
    idx = dist_mtx.argsort()[:, :k]
    labels = [max(list(label_ref[i]), key=list(label_ref[i]).count) for i in idx]
    return labels
    
reference = model.latent[meta["data_type"]=="rna"]
query = model.latent[meta["data_type"]=="atac"]
predict = annotate_by_nn(vec_tar=query, vec_ref=reference, label_ref=np.array(adata_rna.obs["celltype"]))
print(sum(predict==adata_atac.obs["celltype"])/len(predict))
pd.DataFrame(predict).to_csv(os.path.join(args.save_path, "predict.csv"))
pd.DataFrame(adata_atac.obs["celltype"]).to_csv(os.path.join(args.save_path, "query.csv"))



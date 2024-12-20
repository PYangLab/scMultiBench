import os
import time
import umap
import h5py
import random
import anndata
import argparse
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from SMILE.SMILE import SMILE
import torch.nn.functional as F
from SMILE.SMILE import littleSMILE
from SMILE.SMILE import ReferenceSMILE_trainer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer

# The SMILE script for mosaic integration for [RNA, RNA+ATAC, ATAC] data type. The output is a joint embedding (dimensionality reduction).
# run commond for SMILE
# python main_SMILE.py --query_path1   "../../data/dataset_final/D45/rna1.h5"  --query_path2 "../../data/dataset_final/D45/atac3.h5" --ref_path1 "../../data/dataset_final/D45/rna2.h5"  --ref_path2 "../../data/dataset_final/D45/atac2.h5"  --save_path "../../result/embedding/mosaic integration/D45/SMILE/"


parser = argparse.ArgumentParser("SMILE")
parser.add_argument('--ref_path1', metavar='DIR', default='NULL', help='path to train data1')
parser.add_argument('--ref_path2', metavar='DIR', default='NULL', help='path to train data2')
parser.add_argument('--query_path1', metavar='DIR', default='NULL', help='path to train data2')
parser.add_argument('--query_path2', metavar='DIR', default='NULL', help='path to train data2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()
begin_time = time.time()

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()
        
def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.mat(np.array(f['matrix/data']).transpose())
    return X
    
def feature_loader(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/features'])
    return X
    
##Read reference data
ref_rna = data_loader(args.ref_path1)
ref_rna= anndata.AnnData(X=ref_rna)
ref_rna_feature = feature_loader(args.ref_path1)
ref_rna.var_names = ref_rna_feature
print(ref_rna.var_names, "!!!")
ref_rna.obs['source']='RNA-seq'
ref_rna.obs['use']='reference'

ref_dna = data_loader(args.ref_path2)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(ref_dna)
tfidf = tfidf[:,transformer.idf_>=np.percentile(transformer.idf_, 90)]
ref_dna = tfidf.todense()
scaler = StandardScaler()
ref_dna = scaler.fit_transform(ref_dna)
ref_dna= anndata.AnnData(X=ref_dna)
ref_dna.obs['use']='reference'

##Read query data
query_rna = data_loader(args.query_path1)
query_rna= anndata.AnnData(X=query_rna)
query_rna_feature = feature_loader(args.query_path1)
query_rna.var_names = query_rna_feature
print(query_rna.var_names, "@@@")
query_rna.obs['source']='RNA-seq'
query_rna.obs['use']='query'

query_dna = data_loader(args.query_path2)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(query_dna)
tfidf = tfidf[:,transformer.idf_>=np.percentile(transformer.idf_, 90)]
query_dna = tfidf.todense()
scaler = StandardScaler()
query_dna = scaler.fit_transform(query_dna)
query_dna= anndata.AnnData(X=query_dna)
query_dna.obs['use']='query'

sc.pp.normalize_total(ref_rna, target_sum=1e4)
sc.pp.log1p(ref_rna)

sc.pp.normalize_total(query_rna, target_sum=1e4)
sc.pp.log1p(query_rna)

##identify highly variable features across modalities
adata = ref_rna.concatenate(query_rna,query_dna)

sc.pp.highly_variable_genes(ref_rna, n_top_genes=5000,subset=True)
adata_rna = ref_rna.concatenate(query_rna)

#sc.pp.highly_variable_genes(ref_dna, n_top_genes=5000,subset=True)
adata_dna = ref_dna.concatenate(query_dna)

print('total uique cells: '+str(adata.X.shape[0]))
print('# of features in RNA-seq:' + str(adata_rna.X.shape[1]))
print('# of features in ATAC-seq:' + str(adata_dna.X.shape[1]))

##Scale data
X_rna_paired = adata_rna[adata_rna.obs['use']=='reference'].X#.todense()
X_dna_paired = adata_dna[adata_dna.obs['use']=='reference'].X#.todense()
X_rna_unpaired = adata_rna[adata_rna.obs['use']=='query'].X#.todense()
X_dna_unpaired = adata_dna[adata_dna.obs['use']=='query'].X#.todense()

scaler = StandardScaler()
X_rna_paired = scaler.fit_transform(X_rna_paired)

scaler = StandardScaler()
X_rna_unpaired = scaler.fit_transform(X_rna_unpaired)

start_time = time.time()

integrater = littleSMILE(input_dim_a=X_rna_paired.shape[1],input_dim_b=X_dna_paired.shape[1],clf_out=20)
ReferenceSMILE_trainer(X_rna_paired,X_dna_paired,X_rna_unpaired,X_dna_unpaired,integrater,train_epoch=1000)

print("--- %s seconds ---" % int((time.time() - start_time)))

integrater.to(torch.device("cpu"))
integrater.eval()
X_tensor_A=torch.tensor(X_rna_paired).float()
X_tensor_B=torch.tensor(X_dna_paired).float()
X_tensor_uA=torch.tensor(X_rna_unpaired).float()
X_tensor_uB=torch.tensor(X_dna_unpaired).float()

z_a=integrater.encoder_a(X_tensor_A)
z_b=integrater.encoder_b(X_tensor_B)
z_ua=integrater.encoder_a(X_tensor_uA)
z_ub=integrater.encoder_b(X_tensor_uB)

z_a = F.normalize(z_a, dim=1,p=2)
z_b = F.normalize(z_b, dim=1,p=2)
z_a = torch.Tensor.cpu(z_a).detach().numpy()
z_b = torch.Tensor.cpu(z_b).detach().numpy()

z_ua = F.normalize(z_ua, dim=1,p=2)
z_ub = F.normalize(z_ub, dim=1,p=2)
z_ua = torch.Tensor.cpu(z_ua).detach().numpy()
z_ub = torch.Tensor.cpu(z_ub).detach().numpy()

y_pred = np.concatenate(((z_a+z_b)/2,z_ua,z_ub),0)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=(y_pred))
file.close()




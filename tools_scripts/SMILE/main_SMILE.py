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
from SMILE.SMILE import PairedSMILE_trainer
from SMILE.SMILE import Paired_SMILE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer

random.seed(1)
parser = argparse.ArgumentParser("SMILE")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train data1')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train data2')
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
    
rna = data_loader(args.path1)
dna = data_loader(args.path2)

##we use scanpy to preprocess scRNA-seq data 
adata= anndata.AnnData(X=rna)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

##only use highly variable genes as features
sc.pp.highly_variable_genes(adata, min_mean=0.15, max_mean=3, min_disp=0.75)
adata = adata[:,adata.var['highly_variable'].values]

##identify cell-types using scRNA-seq data
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
sc.tl.umap(adata)
sc.tl.leiden(adata,resolution=0.25)
sc.pl.umap(adata, color=['leiden'])

##scaling
rna_X = adata.X
scaler = StandardScaler()
rna_X = scaler.fit_transform(rna_X)

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(dna)
tfidf = tfidf[:,transformer.idf_>=np.percentile(transformer.idf_, 90)]
dna_X = tfidf.todense()
scaler = StandardScaler()
dna_X = scaler.fit_transform(dna_X)

##2. Use mpSMILE for integration
clf_out = 25
net = Paired_SMILE(input_dim_a=rna_X.shape[1],
                         input_dim_b=dna_X.shape[1],clf_out=clf_out)
net.apply(weights_init)

##3. Training process
PairedSMILE_trainer(X_a = rna_X, X_b = dna_X, model = net, num_epoch=20)##training for 20 epochs

##4. Integration visualization
net.to(torch.device("cpu"))
X_all_tensor_a = torch.tensor(rna_X).float()
X_all_tensor_b = torch.tensor(dna_X).float()

y_pred_a = net.encoder_a(X_all_tensor_a)
y_pred_a = F.normalize(y_pred_a, dim=1,p=2)
y_pred_a = torch.Tensor.cpu(y_pred_a).detach().numpy()

y_pred_b = net.encoder_b(X_all_tensor_b)
y_pred_b = F.normalize(y_pred_b, dim=1,p=2)
y_pred_b = torch.Tensor.cpu(y_pred_b).detach().numpy()

y_pred = np.concatenate((y_pred_a, y_pred_b),axis=0)


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=(y_pred))
file.close()


import os
import h5py
import torch
import random
import anndata
import argparse
import scanpy as sc
from network import *
from training import *
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser("sciCAN")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train rna')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train atac')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

# The sciCAN script for diagonal integration requires RNA and ATAC data as input, where ATAC needs to be transformed into gene activity score. The output is a joint embedding (dimensionality reduction).
# run commond for sciCAN
# python main_sciCAN.py --path1 "../../data/dataset_final/D27/rna.h5" --path2 "../../data/dataset_final/D27/atac_gas.h5" --save_path "../../result/embedding/diagonal integration/D27/sciCAN/"

def load_data(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
    return(X)

rna_path = args.path1
rna_X = load_data(rna_path)
adata= anndata.AnnData(X=rna_X)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
rna_X = adata.X
scaler = StandardScaler()
rna_X = scaler.fit_transform(rna_X)

dna_path = args.path2
dna_X = load_data(dna_path)
adata= anndata.AnnData(X=dna_X)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
dna_X = adata.X
scaler = StandardScaler()
dna_X = scaler.fit_transform(dna_X)

FeatureExtractor = Cycle_train_wolabel(epoch=200, batch_size=1024, source_trainset=rna_X, target_trainset=dna_X)

X_tensor_a = torch.tensor(rna_X).float()
X_tensor_b = torch.tensor(dna_X).float()
FeatureExtractor.to(torch.device("cpu"))
X_all_tensor = torch.cat((X_tensor_a,X_tensor_b),0)
y_pred = FeatureExtractor(X_all_tensor)
y_pred = torch.Tensor.cpu(y_pred[2]).detach().numpy()

save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print("create path")
else:
    print("the path exits")
    
file = h5py.File(save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=(y_pred))
file.close()

import os
import time
import h5py
import torch
import random
import anndata
import argparse
import numpy as np
import scanpy as sc
from iPoLNG import iPoLNG

torch.set_default_tensor_type("torch.cuda.FloatTensor" if torch.cuda.is_available() else "torch.FloatTensor")
parser = argparse.ArgumentParser("iPOLNG")
parser.add_argument('--path1', metavar='DIR',  default="", help='path to RNA')
parser.add_argument('--path2', metavar='DIR',  default="", help='path to ATAC')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

# The iPOLNG script for vertical integration requires RNA and ATAC data as input. The output is a joint embedding (dimensionality reduction).
# run commond for iPOLNG
# python main_iPOLNG.py --path1  "../../data/dataset_final/D15/rna.h5" --path2 "../../data/dataset_final/D15/atac.h5"  --save_path "../../result/embedding/vertical integration/iPOLNG/D15/"

begin_time = time.time()
def data_loader(path, top_number):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
        adata= anndata.AnnData(X=X)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=top_number)
        selected_features = adata.var['highly_variable'].values
        X = X[:,selected_features]
        X = torch.from_numpy(X)
    return X
    
def run_iPOLNG(data1_path, data2_path):
    data1 = data_loader(data1_path, 5000)
    data2 = data_loader(data2_path, 20000)
    print(data1.shape)
    print(data2.shape)
    W = {"W1":data1.type("torch.cuda.FloatTensor"), "W2":data2.type("torch.cuda.FloatTensor")}
    model = iPoLNG.iPoLNG(W, num_topics=20, integrated_epochs=3000, warmup_epochs=3000, seed=42, verbose=True)
    result = model.Run()
    embedding = result['L_est']
    embedding_rna = result['Ls_est']['W1']
    embedding_atac = result['Ls_est']['W2']
    return embedding, embedding_rna, embedding_atac

embedding, embedding_rna, embedding_atac = run_iPOLNG(args.path1, args.path2)
end_time = time.time()
all_time = end_time - begin_time

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=embedding)
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")

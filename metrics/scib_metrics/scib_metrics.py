import scanpy as sc
import scib
import h5py
import numpy as np
import anndata as ad
import pandas as pd
import argparse
import os


###
parser = argparse.ArgumentParser("scib")
parser.add_argument('--data_path', metavar='DIR', nargs='+',default=[], help='path to train data')
parser.add_argument('--cty_path', metavar='DIR', nargs='+', default= [], help='path to train data2')
parser.add_argument('--save_path', metavar='DIR', default="", help='path to save the output data')
parser.add_argument('--transpose', type=int, default= 1, help='')
args = parser.parse_args()

if args.transpose==1:
    transpose = True
else:
    transpose = False
    

def h5_to_matrix_integrated(data_paths):
    data = []
    for path in data_paths:
        with h5py.File(path, "r") as f:
            print(args.transpose)
            if transpose:
                X = np.mat(np.array(f['data']).transpose())
            else:
                X = np.mat(np.array(f['data']))
            print(X.shape)
            print(X.transpose().shape)
        data.append(X)
    data = np.concatenate(data,0)
    return data
    
def read_label(label_paths):
    all_labels = []
    for path in label_paths:
        label_fs = pd.read_csv(path, header=None, index_col=False)
        label_fs = label_fs.iloc[1:(label_fs.shape[0]), 1]
        all_labels.append(label_fs)
    all_labels = pd.concat(all_labels)
    all_labels = pd.Categorical(all_labels).codes
    all_labels = np.array(all_labels[:]).astype('int32')
    return all_labels
    
def read_batch(label_paths):
    all_batchs = []
    i = 0
    for path in label_paths:
        label_fs = pd.read_csv(path, header=None, index_col=False)
        label_fs = label_fs.iloc[1:(label_fs.shape[0]), 1]
        batch_temp = np.ones(len(label_fs))+i
        i = i + 1
        all_batchs.append(batch_temp)
    all_batchs = np.concatenate(all_batchs)
    return all_batchs
    
cty = read_label(args.cty_path)
batch = read_batch(args.cty_path)

adata_integrated = h5_to_matrix_integrated(args.data_path)
adata_integrated = ad.AnnData(adata_integrated)
adata_integrated.obsm["X_emb"] = adata_integrated.X
adata_integrated.obs["celltype"] = cty
adata_integrated.obs['celltype'] = adata_integrated.obs['celltype'].astype(str).astype('category')
adata_integrated.obs["batch"] = batch
adata_integrated.obs['batch'] = adata_integrated.obs['batch'].astype(str).astype('category')
sc.pp.neighbors(adata_integrated)
adata_unintegrated = adata_integrated
metrics = scib.metrics.metrics(
    adata_unintegrated,
    adata_integrated,
    batch_key='batch',
    label_key= 'celltype',
    embed='X_emb',
    ari_=True,
    nmi_=True,
    silhouette_=True,
    graph_conn_= True,
    #pcr_=True,
    kBET_=True,
    isolated_labels_asw_=True,
    isolated_labels_f1_= True,
    lisi_graph_ = True
)
result = metrics[metrics.notna().any(axis=1)]
print(result)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")

result.to_csv(args.save_path+"/metric.csv", index=True)

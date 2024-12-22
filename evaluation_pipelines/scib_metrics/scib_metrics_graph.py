import os
import scib
import h5py
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from umap.umap_ import fuzzy_simplicial_set

parser = argparse.ArgumentParser("scib")
parser.add_argument('--data_path', metavar='DIR', default="", help='path to train data')
parser.add_argument('--knn_indices', metavar='DIR', default="", help='path to train data')
parser.add_argument('--knn_dists', metavar='DIR', default="", help='path to train data')
parser.add_argument('--cty_path', metavar='DIR', nargs='+', default= [], help='path to train data2')
parser.add_argument('--cluster_path', metavar='DIR', default= "./cluster.h5", help='path to train data')
parser.add_argument('--batch_cluster_path', metavar='DIR', default= "./cluster_batch.h5", help='path to train data')
parser.add_argument('--save_path', metavar='DIR', default="./", help='path to save the output data')
args = parser.parse_args()

#python scib_metrics_graph.py --knn_indices "./../../data/dr&bc/graph/knn_indices.h5" --knn_dists "./../../data/dr&bc/graph/knn_dists.h5"  --cty_path "../../data/dr&bc/graph/cty1.csv"  "../../data/dr&bc/graph/cty2.csv"    --cluster_path "../../data/clustering/graph/sinfonia_clustering.h5" --batch_cluster_path "../../data/clustering/graph/sinfonia_clustering_batch.h5" --save_path "./"

def h5_to_matrix_integrated(path):
    with h5py.File(path, "r") as f:
        X = np.mat(np.array(f['data']))
    if X.shape[0] < X.shape[1]:
        X = X.transpose()
    return X
    
def h5_to_clustering(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['/obs/cluster_leiden'])
        X = np.array([x.decode('utf-8') for x in X.flatten()]).astype(int)
    return X
    
def read_label(label_paths):
    all_labels = []
    for path in label_paths:
        label_fs = pd.read_csv(path, header=None, index_col=False)
        label_fs = label_fs.iloc[1:(label_fs.shape[0]), 0]
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
        label_fs = label_fs.iloc[1:(label_fs.shape[0]), 0]
        batch_temp = np.ones(len(label_fs))+i
        i = i + 1
        all_batchs.append(batch_temp)
    all_batchs = np.concatenate(all_batchs)
    return all_batchs
    
knn_indices = h5_to_matrix_integrated(args.knn_indices)
knn_dists = h5_to_matrix_integrated(args.knn_dists)
print(knn_indices.shape)
print(knn_dists.shape)
cty = read_label(args.cty_path)
batch = read_batch(args.cty_path)
num = max(batch)+1

X = coo_matrix(([], ([], [])), shape=(knn_indices.shape[0], 1))
n_neighbors = 100
set_op_mix_ratio=1.0
local_connectivity=1.0
connectivities = fuzzy_simplicial_set(
    X,
    n_neighbors,
    None,
    None,
    knn_indices=knn_indices,
    knn_dists=knn_dists,
    set_op_mix_ratio=set_op_mix_ratio,
    local_connectivity=local_connectivity,
)
connectivities = connectivities[0]
neighbors = np.zeros((connectivities.shape[0], connectivities.shape[1]), dtype=int)
for i in range(connectivities.shape[0]):
    indices = np.argsort(connectivities[i])[-n_neighbors:]
    neighbors[i, indices] = 1
neighbors = csr_matrix(neighbors)
adata_integrated = ad.AnnData(np.random.rand(neighbors.shape[0],100))
adata_integrated.uns['neighbors'] = {}
adata_integrated.obsp['neighbors'] = neighbors
adata_integrated.obsp['connectivities'] = connectivities

adata_integrated.obs["celltype"] = cty
adata_integrated.obs["celltype"] = adata_integrated.obs['celltype'].astype('category')
adata_integrated.obs["cluster"] = h5_to_clustering(args.cluster_path)
adata_integrated.obs["cluster"] = adata_integrated.obs['cluster'].astype('category')
adata_integrated.obs["batch_cluster"] = h5_to_clustering(args.batch_cluster_path)
adata_integrated.obs["batch_cluster"] = adata_integrated.obs['batch_cluster'].astype('category')
adata_integrated.obs["batch"] = batch
adata_integrated.obs["batch"] = adata_integrated.obs['batch'].astype('category')

clisi = scib.me.clisi_graph(adata_integrated, label_key="celltype", type_="knn")
print("cLISI:", clisi)

ari = scib.me.ari(adata_integrated, cluster_key="cluster", label_key="celltype")
print("ARI_cellType:", ari)
nmi = scib.me.nmi(adata_integrated, cluster_key="cluster", label_key="celltype")
print("NMI_cellType:", nmi)
ifi = scib.me.isolated_labels_f1(adata_integrated, batch_key="batch", label_key="celltype", cluster_key="cluster", embed=None, iso_threshold=num)
print("iF1:", ifi)


ari_batch = 1 - abs(scib.me.ari(adata_integrated, cluster_key="batch_cluster", label_key="batch"))
print("ARI_batch:", ari_batch)
nmi_batch = 1 - abs(scib.me.nmi(adata_integrated, cluster_key="batch_cluster", label_key="batch"))
print("NMI_batch:", nmi_batch)
gc = scib.me.graph_connectivity(adata_integrated, label_key="celltype")
print("GC:", gc)
ilisi = scib.me.ilisi_graph(adata_integrated, batch_key="batch", type_="knn")
print("iLISI:", ilisi)

adata_integrated.obs['batch'] = adata_integrated.obs['batch'].astype(str)
adata_integrated.obs['celltype'] = adata_integrated.obs['celltype'].astype(str)
kbet = scib.me.kBET(adata_integrated, batch_key="batch", label_key="celltype", type_="knn")
print("kbet:", kbet)

results_dict = {
  'ARI': ari,
  'NMI': nmi,
  'iF1': ifi,
  'cLISI': clisi,
  'GC': gc,
  'iLISI': ilisi,
  'kBET': kbet
  }
result = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Value'])
print(result)
if not os.path.exists(args.save_path):
  os.makedirs(args.save_path)
  print("create path")
else:
  print("the path exits")
result.to_csv(args.save_path+"/metric.csv", index=True)

import os
import scib
import h5py
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad

parser = argparse.ArgumentParser("scib")
parser.add_argument('--data_path', metavar='DIR', nargs='+',default=[], help='path to embedding')
parser.add_argument('--cty_path', metavar='DIR', nargs='+', default= [], help='path to cell type')
parser.add_argument('--cluster_path', metavar='DIR', default= "./cluster.h5", help='path to cell type clustering path')
parser.add_argument('--batch_cluster_path', metavar='DIR', default= "./cluster.h5", help='path to batch clustering path')
parser.add_argument('--save_path', metavar='DIR', default="./", help='path to save the output data')
args = parser.parse_args()

# python scib_metrics.py --data_path "../../data/dr&bc/embedding/embedding.h5" --cty_path "../../data/dr&bc/embedding/cty1.csv"  "../../data/dr&bc/embedding/cty2.csv"  "../../data/dr&bc/embedding/cty3.csv" --cluster_path "../../data/clustering/embedding/sinfonia_clustering.h5" --batch_cluster_path "../../data/clustering/embedding/sinfonia_clustering_batch.h5" --save_path "./"

def h5_to_matrix_integrated(data_paths):
    data = []
    for path in data_paths:
        with h5py.File(path, "r") as f:
            X = np.asarray(np.mat(np.array(f['data'])))
            if X.shape[0] < X.shape[1]:
                X = X.transpose()
        data.append(X)
    data = np.concatenate(data,0)
    return data
    
def h5_to_matrix_integrated_ori(data_paths):
    data = []
    for path in data_paths:
        with h5py.File(path, "r") as f:
            X = np.asarray(np.mat(np.array(f['matrix/data'])))
            X = X.transpose()
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
    
def h5_to_clustering(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['/obs/cluster_leiden'])
        X = np.array([x.decode('utf-8') for x in X.flatten()]).astype(int)
    return X
    
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
num = np.max(batch)+1
print(num)
print(cty)
adata_integrated = h5_to_matrix_integrated(args.data_path)
adata_integrated = ad.AnnData(adata_integrated)
adata_integrated.obsm["X_emb"] = adata_integrated.X
adata_integrated.obs["celltype"] = cty
adata_integrated.obs["celltype"] = adata_integrated.obs['celltype'].astype(str).astype('category')
adata_integrated.obs["cluster"] = h5_to_clustering(args.cluster_path)
adata_integrated.obs["cluster"] = adata_integrated.obs['cluster'].astype('category')
adata_integrated.obs["batch_cluster"] = h5_to_clustering(args.batch_cluster_path)
adata_integrated.obs["batch_cluster"] = adata_integrated.obs['batch_cluster'].astype('category')
adata_integrated.obs["batch"] = batch
adata_integrated.obs["batch"] = adata_integrated.obs['batch'].astype(str).astype('category')

clisi = scib.me.clisi_graph(adata_integrated, label_key="celltype", type_="embed", use_rep="X_emb")
print("clisi:", clisi)

ari = scib.me.ari(adata_integrated, cluster_key="cluster", label_key="celltype")
print("ARI_cellType:", ari)
nmi = scib.me.nmi(adata_integrated, cluster_key="cluster", label_key="celltype")
print("NMI_cellType:", nmi)
asw = scib.me.silhouette(adata_integrated, label_key="cluster", embed="X_emb")
print("ASW_cellType:", asw)
iasw = scib.me.isolated_labels_asw(adata_integrated, batch_key="batch", label_key="cluster", embed="X_emb", iso_threshold=num)
print("iASW:", iasw)
if1 = scib.me.isolated_labels_f1(adata_integrated, batch_key="batch", label_key="celltype", cluster_key="cluster", embed="X_emb", iso_threshold=num)
print("iF1:", if1)

sc.pp.neighbors(adata_integrated, use_rep="X_emb")
asw_batch = scib.me.silhouette_batch(adata_integrated, batch_key="batch", label_key="celltype", embed="X_emb")
print("ASW_batch:", asw_batch)
gc = scib.me.graph_connectivity(adata_integrated, label_key="celltype")
print("GC:", gc)
ilisi = scib.me.ilisi_graph(adata_integrated, batch_key="batch", type_="embed", use_rep="X_emb")
print("iLISI:", ilisi)
ari_batch = 1 - abs(scib.me.ari(adata_integrated, cluster_key="batch_cluster", label_key="batch"))
print("ARI_batch:", ari_batch)
nmi_batch = 1 - abs(scib.me.nmi(adata_integrated, cluster_key="batch_cluster", label_key="batch"))
print("NMI_batch:", nmi_batch)
kbet = scib.me.kBET(adata_integrated, batch_key="batch", label_key="celltype", type_="embed", embed="X_emb")
print("kbet:", kbet)

results_dict = {
    'cLISI': clisi,
    'ARI': ari,
    'NMI': nmi,
    'ASW': asw,
    'iASW': iasw,
    'iF1': if1,
    'ASW_batch': asw_batch,
    'ARI_batch': ari_batch,
    'NMI_batch': nmi_batch,
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


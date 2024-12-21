# %%
import os
import h5py
import sinfonia
import warnings
import argparse
import numpy as np
import scanpy as sc
import pandas as pd

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("SINFONIA")
parser.add_argument('--path1', metavar='DIR', type=str, required=True, help='Path to the embedding h5 file')
parser.add_argument('--cty_path', metavar='DIR', nargs='+', default=[], help='Path to the cty CSV file')
parser.add_argument('--save_path', metavar='DIR', type=str, required=True, help='Path to save the output h5 file')
args = parser.parse_args()

# python SINFONIA.py --path1 "../../data/dr&bc/embedding/embedding.h5" --cty_path "../../data/dr&bc/embedding/cty1.csv" "../../data/dr&bc/embedding/cty2.csv" "../../data/dr&bc/embedding/cty3.csv" --save_path "./../../data/clustering/embedding/sinfonia_clustering.h5"

def h5_to_anndata(data_path, cty_path):
    with h5py.File(data_path, 'r') as f:
        data_data = np.array(f['data'])
        if data_data.shape[0] < data_data.shape[1]:
            data_data = data_data.T
    print(np.sum(np.isnan(data_data)))
    data_data[np.isnan(data_data)] = 0
    adata = sc.AnnData(X=data_data)
    combined_cty = pd.concat([pd.read_csv(path) for path in cty_path], ignore_index=True)
    adata.obs['label'] = combined_cty['x'].values
    return adata

# %%
def sinfonia_process(data_path,cty_path,save_dir):
    sinfonia.setup_seed(2022)
    adata = h5_to_anndata(data_path, cty_path)
    print(adata.obs['label'])
    sc.pp.neighbors(adata)
    adata = sinfonia.get_N_clusters(adata, n_cluster=adata.obs['label'].nunique(), cluster_method='leiden')
    adata.obs['cluster_leiden'] = adata.obs['leiden']
    save_path = os.path.dirname(save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")
    else:
        print(f"Directory already exists: {save_path}")
        
    with h5py.File(save_dir, 'w') as f:
        f.create_dataset('X', data=adata.X)
        for col in adata.obs.columns:
            f.create_dataset(f'obs/{col}', data=adata.obs[col].astype('S'))  
        for col in adata.var.columns:
            f.create_dataset(f'var/{col}', data=adata.var[col].astype('S'))  
        f.create_dataset('obs_names', data=adata.obs_names.to_numpy().astype('S'))  
        f.create_dataset('var_names', data=adata.var_names.to_numpy().astype('S'))  

    print(f"Data has been successfully saved to {save_dir}")
    return adata

# %%
sinfonia_process(args.path1,args.cty_path,args.save_path)






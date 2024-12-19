# %%
import os
import ot
import glob
import math
import time
import torch
import scipy
import random
import sklearn
import anndata
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from gpsa import rbf_kernel
from gpsa import VariationalGPSA
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

# %%
parser = argparse.ArgumentParser('GPSA')
parser.add_argument('--data_dir', default='../unified_data/DLPFC/donor1/', help='path to the data directory')
parser.add_argument('--save_dir', default='./aligned_slices/', help='path to save the output data')
args = parser.parse_args()

# The GPSA script for cross-integration requires spatial data in 'h5ad' format as input, including both gene expression data and spatial coordinates. The output is aligned coordinates (spatial registration).
# run commond for GPSA
# python GPSA.py --data_dir '../unified_data/DLPFC/donor1/' --save_dir './aligned_slices/'

# %%
def load_slices_h5ad(data_dir):
    slices = []
    file_paths = glob.glob(data_dir + "*.h5ad")
    for file_path in file_paths:
        slice_i = sc.read_h5ad(file_path)
        
        if scipy.sparse.issparse(slice_i.X):
            slice_i.X = slice_i.X.toarray()
        
        Ground_Truth = slice_i.obs['Ground_Truth']
        slice_i.obs = pd.DataFrame({'Ground_Truth': Ground_Truth})
        slices.append(slice_i)
    
    return slices

# %%
# https://github.com/andrewcharlesjones/spatial-alignment/blob/main/experiments/expression/st/st_alignment.py
def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    # adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # sc.pp.filter_cells(adata, min_counts=100)
    # sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata
# https://github.com/andrewcharlesjones/spatial-alignment/blob/main/experiments/expression/st/st_alignment.py
def process1(N_GENES,data_dir,n_views):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    slices = load_slices_h5ad(data_dir)
    use_gpu=True
    processed_slices = []
    for slice_data in slices:
        processed_data = process_data(slice_data, n_top_genes=3000)
        processed_slices.append(processed_data)
    ## Save original data
    plt.figure(figsize=(20, 6))

    # Add a 'batch' column to each AnnData object
    for i, slice_i in enumerate(processed_slices):
        slice_i.obs['batch'] = int(i)

    # Concatenate the AnnData objects
    #only keep the shared genes
    data = anndata.concat(processed_slices, merge='unique', index_unique='-')
    shared_gene_names = data.var.index.values
    data_knn = processed_slices[1][:, shared_gene_names]
    X_knn = data_knn.obsm["spatial"]
    Y_knn = data_knn.X
    Y_knn = (Y_knn - Y_knn.mean(0)) / Y_knn.std(0)
    # nbrs = NearestNeighbors(n_neighbors=2).fit(X_knn)
    # distances, indices = nbrs.kneighbors(X_knn)
    knn = KNeighborsRegressor(n_neighbors=10, weights="uniform").fit(X_knn, Y_knn)
    preds = knn.predict(X_knn)
    r2_vals = r2_score(Y_knn, preds, multioutput="raw_values")

    gene_idx_to_keep = np.where(r2_vals > 0.3)[0]
    N_GENES = min(N_GENES, len(gene_idx_to_keep))
    gene_names_to_keep = data_knn.var.index.values[gene_idx_to_keep]
    gene_names_to_keep = gene_names_to_keep[np.argsort(-r2_vals[gene_idx_to_keep])]
    r2_vals_sorted = -1 * np.sort(-r2_vals[gene_idx_to_keep])
    if N_GENES < len(gene_names_to_keep):
        gene_names_to_keep = gene_names_to_keep[:N_GENES]
    data = data[:, gene_names_to_keep]
    n_samples_list = [slice_.shape[0] for slice_ in processed_slices]
    cumulative_sum = np.cumsum(n_samples_list)
    cumulative_sum = np.insert(cumulative_sum, 0, 0)
    view_idx = [
        np.arange(cumulative_sum[ii], cumulative_sum[ii + 1]) for ii in range(n_views)
    ]

    X_list = []
    Y_list = []
    for vv in range(n_views):
        curr_X = np.array(data[data.obs.batch == vv].obsm["spatial"])
        curr_Y = data[data.obs.batch == vv].X

        curr_X = scale_spatial_coords(curr_X)
        curr_Y = (curr_Y - curr_Y.mean(0)) / curr_Y.std(0)

        X_list.append(curr_X)
        Y_list.append(curr_Y)


    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    x = torch.from_numpy(X).float().clone().to(device)
    y = torch.from_numpy(Y).float().clone().to(device)
    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }
    return x,slices,data_dict , data

# %%
def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val

# %%
def train(model, loss_fn, optimizer,x,view_idx,Ns,data_dict):
    model.train()

    # Forward pass
    G_means, G_samples, F_latent_samples, F_samples = model.forward(
        X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns, S=5
    )

    # Compute loss
    loss = loss_fn(data_dict, F_samples)
    # Compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), G_means

# https://github.com/andrewcharlesjones/spatial-alignment/blob/main/experiments/expression/st/st_alignment.py
def whole_process(data_dir,save_dir,num_slices):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N_GENES = 20
    N_SAMPLES = None
    N_LAYERS = num_slices
    fixed_view_idx = 1

    n_spatial_dims = 2
    n_views = num_slices
    m_G = 200
    m_X_per_view = 200

    N_LATENT_GPS = {"expression": None}
    ###############################################
    N_EPOCHS = 5000
    PRINT_EVERY = 25
    #x, slices,data_dict,data = process1(N_GENES,data_dir,file_names,num_slices,mapping_dict)
    x, slices,data_dict,data = process1(N_GENES,data_dir,n_views)
    model = VariationalGPSA(
    data_dict,
    n_spatial_dims=n_spatial_dims,
    m_X_per_view=m_X_per_view,
    m_G=m_G,
    data_init=True,
    minmax_init=False,
    grid_init=False,
    n_latent_gps=N_LATENT_GPS,
    mean_function="identity_fixed",
    kernel_func_warp=rbf_kernel,
    kernel_func_data=rbf_kernel,
    fixed_view_idx=fixed_view_idx,
    ).to(device)
    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for t in range(N_EPOCHS):
        loss, G_means =  train(model, model.loss_fn, optimizer,x,view_idx,Ns,data_dict)
        if t % PRINT_EVERY == 0:
                print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
                curr_aligned_coords = G_means["expression"].detach().cpu().numpy()
        
                if model.n_latent_gps["expression"] is not None:
                    curr_W = model.W_dict["expression"].detach().numpy()
                    pd.DataFrame(curr_W).to_csv("./out/W_st.csv")

    # Convert the tensor to a numpy array on the CPU
    G_means_expression = G_means['expression'].cpu().detach().numpy()

    # Iterate through each slice and update its spatial coordinates
    for i, slice_i in enumerate(slices):
        # Get the indices of data points belonging to this slice
        slice_indices = np.where(data.obs['batch'] == i)[0]

        # Update the spatial coordinates of this slice with the corresponding aligned coordinates
        slice_i.obsm['spatial'] = G_means_expression[slice_indices, :]

    original_slices = load_slices_h5ad(data_dir)

    for original_slice, updated_slice in zip(original_slices, slices):
        original_slice.obsm['spatial'] = updated_slice.obsm['spatial']
    return original_slices

# %%
def combine(data_dir,save_dir):
    begin_time = time.time()
    slices = load_slices_h5ad(data_dir)
    num_slices=len(slices)
    slices_coordinated = whole_process(data_dir,save_dir,num_slices)
    end_time = time.time()
    all_time = end_time - begin_time
    directory = os.path.dirname(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("create path")
    else:
        print("the path exits")
    np.savetxt(args.save_dir, [all_time], delimiter=",")
    return slices_coordinated

aligned_slices = combine(args.data_dir, args.save_dir)

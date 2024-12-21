# %%
import numpy as np
import pandas as pd
import h5py
import scanpy as sc
import sinfonia
import warnings
import igraph
import argparse
import os
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph, NearestNeighbors
warnings.filterwarnings("ignore")

# %%
parser = argparse.ArgumentParser("SINFONIA_SCMOMAT")
parser.add_argument('--knn_dist_path', metavar='str',  default=[], help='distances')
parser.add_argument('--knn_indices_path', metavar='str',  default=[], help='connectivities')
parser.add_argument('--cty_path', metavar='DIR', nargs='+', default=[], help='Path to the cty CSV file')
parser.add_argument('--save_path', metavar='DIR', type=str, required=True, help='Path to save the output h5 file')
parser.add_argument('--num',  type=int,  default=1, help='path to save the output data')
args = parser.parse_args()

# python SINFONIA_graph.py --knn_dist_path "../../data/dr&bc/graph/knn_dists.h5" --knn_indices_path "../../data/dr&bc/graph/knn_indices.h5"  --num 2  --save_path "./sinfonia_clustering_batch.h5"

def compute_connectivities_umap(
    knn_indices,
    knn_dists,
    n_neighbors,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
):
    """\
    This is from umap.fuzzy_simplicial_set [McInnes18]_.
    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """

    from umap.umap_ import fuzzy_simplicial_set
    from scipy.sparse import coo_matrix

    # place holder since we use precompute matrix
    X = coo_matrix(([], ([], [])), shape=(knn_indices.shape[0], 1))
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

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    return connectivities.tocsr()

# %%
def sinfonia_process_scmomat(knn_dist_path,knn_indices_path,num,save_dir):
    sinfonia.setup_seed(2022)
    with h5py.File(knn_dist_path, 'r') as f:
        distances = np.array(f['data'])  
    with h5py.File(knn_indices_path, 'r') as f:
        indices = np.array(f['data']).astype(int) 
    if indices.shape[0] < indices.shape[1]:
        indices = np.transpose(indices)
        distances = np.transpose(distances)

    print(indices.shape,distances.shape,"!!!!")
    print(distances.dtype, "@@@@")
    print(indices.dtype, "!!!!")

    # cty = pd.read_csv(cty_path)
    # cty = cty.values
    n_cells = indices.shape[0]
    print(n_cells,"@@")
    knn_graph = np.zeros((n_cells, n_cells))
    print(indices,"indices")
    knn_graph[np.arange(n_cells)[:, None], indices] = distances

    connectivities = compute_connectivities_umap(indices, distances, n_neighbors=30)
    distances_sparse = sp.csr_matrix(knn_graph)
    n_cells = indices.shape[0]
    adata = sc.AnnData(obs=pd.DataFrame(index=np.arange(n_cells)))
    n_neighbors = 30
    adata.obsp['connectivities'] = connectivities
    adata.obsp['distances'] = distances_sparse
    adata.uns['neighbors'] = {}
    adata.uns['neighbors']['connectivities_key'] = 'connectivities'
    adata.uns['neighbors']['distances_key'] = 'distances'
    adata.uns['neighbors']['params'] = {
        'n_neighbors': n_neighbors,  
        'method': 'umap',   
        'metric': 'euclidean'  
    }
        
    adata = sinfonia.get_N_clusters(adata, n_cluster=num, cluster_method='leiden')
    adata.obs['cluster_leiden'] = adata.obs['leiden']

    save_path = os.path.dirname(save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")
    else:
        print(f"Directory already exists: {save_path}")
        
    with h5py.File(save_dir, 'w') as f:
        zero_matrix = np.zeros((n_cells, 1))
        f.create_dataset('X', data=zero_matrix)
        print("adata.X is empty; filled with zero matrix.")
        for col in adata.obs.columns:
            f.create_dataset(f'obs/{col}', data=adata.obs[col].astype('S'))  
        for col in adata.var.columns:
            f.create_dataset(f'var/{col}', data=adata.var[col].astype('S'))  

    print(f"Data has been successfully saved to {save_dir}")
    return adata

sinfonia_process_scmomat(args.knn_dist_path,args.knn_indices_path,args.num,args.save_path)




import ot
import os
import glob
import math
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
from scipy.spatial import distance_matrix
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

parser = argparse.ArgumentParser('GPSA')
parser.add_argument('--data_dir', default='../unified_data/DLPFC/donor1/', help='path to the data directory')
parser.add_argument('--save_dir', default='./aligned_slices/', help='path to save the output data')
args = parser.parse_args()

# run commond for GPSA
# python main_GPSA.py --data_dir '../unified_data/DLPFC/donor1/' --save_dir './aligned_slices/'


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

###################### Metric 1 PAA ##############################
def create_binary_matrix(slice, n_categories):
    binary_matrix = np.zeros((slice.n_obs, n_categories))
    for idx, cat in enumerate(slice.obs['Ground_Truth'].cat.codes):
        binary_matrix[idx, cat] = 1
    return binary_matrix

def calculate_PAA(slices, n_categories):
    total_accuracy = 0
    num_pairs = 0  
    for i in range(len(slices)):
        for j in range(i + 1, len(slices)):  
            binary_matrix_i = create_binary_matrix(slices[i], n_categories)
            binary_matrix_j = create_binary_matrix(slices[j], n_categories)
            matched_pairs = np.dot(binary_matrix_i, binary_matrix_j.T)
            Z = slices[i].obsm['spatial']
            Z_prime = slices[j].obsm['spatial']
            cost_matrix = ot.dist(Z, Z_prime, metric='euclidean')
            ot_plan = ot.emd([], [], cost_matrix)
            total_accuracy += np.sum(ot_plan * matched_pairs)
            num_pairs += 1
    ave_accuracy = total_accuracy / num_pairs  
    print(ave_accuracy)
    return ave_accuracy
###################### Metric 1 PAA ##############################




###################### Metric 2 SCS ##############################
def create_graph(adata, degree = 4):
        """
        Converts spatial coordinates into graph using networkx library.

        param: adata - ST Slice
        param: degree - number of edges per vertex

        return: 1) G - networkx graph
                2) node_dict - dictionary mapping nodes to spots
        """
        D = distance_matrix(adata.obsm['spatial'], adata.obsm['spatial'])
        # Get column indexes of the degree+1 lowest values per row
        idx = np.argsort(D, 1)[:, 0:degree+1]
        # Remove first column since it results in self loops
        idx = idx[:, 1:]

        G = nx.Graph()
        for r in range(len(idx)):
            for c in idx[r]:
                G.add_edge(r, c)

        node_dict = dict(zip(range(adata.shape[0]), adata.obs.index))
        return G, node_dict

def generate_graph_from_labels(adata, labels_dict):
    """
    Creates and returns the graph and dictionary {node: cluster_label} for specified layer
    """

    g, node_to_spot = create_graph(adata)
    spot_to_cluster = labels_dict

    # remove any nodes that are not mapped to a cluster
    removed_nodes = []
    for node in node_to_spot.keys():
        if (node_to_spot[node] not in spot_to_cluster.keys()):
            removed_nodes.append(node)

    for node in removed_nodes:
        del node_to_spot[node]
        g.remove_node(node)

    labels = dict(zip(g.nodes(), [spot_to_cluster[node_to_spot[node]] for node in g.nodes()]))
    return g, labels


def spatial_entropy(g, labels):
    """
    Calculates spatial entropy of graph
    """
    # construct contiguity matrix C which counts pairs of cluster edges
    cluster_names = np.unique(list(labels.values()))
    C = pd.DataFrame(0,index=cluster_names, columns=cluster_names)

    for e in g.edges():
        C[labels[e[0]]][labels[e[1]]] += 1

    # calculate entropy from C
    C_sum = C.values.sum()
    H = 0
    for i in range(len(cluster_names)):
        for j in range(i, len(cluster_names)):
            if (i == j):
                z = C[cluster_names[i]][cluster_names[j]]
            else:
                z = C[cluster_names[i]][cluster_names[j]] + C[cluster_names[j]][cluster_names[i]]
            if z != 0:
                H += -(z/C_sum)*math.log(z/C_sum)
    return H



def spatial_coherence_score(graph, labels):
    g, l = graph, labels
    true_entropy = spatial_entropy(g, l)
    entropies = []
    for i in range(1000):
        new_l = list(l.values())
        random.shuffle(new_l)
        labels = dict(zip(l.keys(), new_l))
        entropies.append(spatial_entropy(g, labels))

    return abs((true_entropy - np.mean(entropies))/np.std(entropies))


def average_spatial_coherence_score(aligned_slices):
    total_score = 0
    num_slices = len(aligned_slices)
    for aligned_slice in aligned_slices:
        labels_dict = dict(zip(aligned_slice.obs.index, aligned_slice.obs['Ground_Truth']))
        g, labels = generate_graph_from_labels(aligned_slice, labels_dict)

        score = spatial_coherence_score(g, labels)
        total_score += score

    average_spatial_coherence_score = total_score / num_slices
    print("Average Spatial Coherence Score:", average_spatial_coherence_score)
    return average_spatial_coherence_score
###################### Metric 2 SCS ##############################


###################### Metric 3 LTARI #############################
def compute_average_ltari(slices, k=1):
    average_ltari_all_slices = []
    for ref_index in range(len(slices)):
        reference_slice = slices[ref_index]
        ref_coords = reference_slice.obsm['spatial']
        nn_model = NearestNeighbors(n_neighbors=k).fit(ref_coords)
        ltari_values = []
        for query_index in range(len(slices)):
            if query_index != ref_index:
                query_slice = slices[query_index]
                query_coords = query_slice.obsm['spatial']
                _, nearest_indices = nn_model.kneighbors(query_coords)
                if k == 1:
                    transferred_labels = reference_slice.obs['Ground_Truth'].iloc[nearest_indices.flatten()].values
                else:
                    transferred_labels = np.array([reference_slice.obs['Ground_Truth'].iloc[indices].mode()[0] for indices in nearest_indices])
                ari = adjusted_rand_score(query_slice.obs['Ground_Truth'], transferred_labels)
                ltari_values.append(ari)
        average_ltari_all_slices.append(np.mean(ltari_values))
    final_average_ltari = np.mean(average_ltari_all_slices)
    return final_average_ltari

###################### Metric 3 LTARI #############################

def combine(data_dir,save_dir):
    
    # file_names = [f for f in os.listdir(data_dir) if f.endswith('.h5ad') and os.path.isfile(os.path.join(data_dir, f))]
    slices = load_slices_h5ad(data_dir)
    unique_layers = set()
    
    for slice in slices:
        unique_layers.update(slice.obs['Ground_Truth'].unique())

    n_labels = len(unique_layers)
    #EVALUATION
    print("PAA of this model is:")
    PAA = calculate_PAA(slices,n_labels)
    print("LTARI of this model is:")
    compute_average_ltari_result = compute_average_ltari(slices)
    print(compute_average_ltari_result)
    print("SCS of this model is:")
    SCS = average_spatial_coherence_score(slices)

    metrics_data = [
            {"Metric": "PAA", "Value": PAA},
            {"Metric": "SCS", "Value": SCS},
            {"Metric": "LTARI", "Value": compute_average_ltari_result}
        ]

    metrics_df = pd.DataFrame(metrics_data)
    data_dir_name = os.path.basename(os.path.normpath(data_dir))
    csv_filename = f"{data_dir_name}_metrics.csv"

    # Ensure the directory exists
    save_path = os.path.join(save_dir, csv_filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the DataFrame to CSV
    metrics_df.to_csv(save_path, index=False)

combine(args.data_dir,args.save_dir)
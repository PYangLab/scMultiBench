import os
import ot
import math
import time
import glob
import scipy
import random
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import paste as pst
import networkx as nx
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

parser = argparse.ArgumentParser('paste')
parser.add_argument('--data_dir', default='../unified_data/SCC/patient_2/', help='path to the data directory')
parser.add_argument('--save_dir', default='./aligned_slices/', help='path to save the output data')
args = parser.parse_args()

#python PASTE_pairwise.py --data_dir '../../data/dataset_final/D60/processed/patient_2/' --save_dir '../../result/registration/D60/paste/patient_2'

def load_slices_h5ad_scc(data_dir):
    slices = []
    file_paths = glob.glob(data_dir + "*.h5ad")
    for file_path in file_paths:
        slice_i = sc.read_h5ad(file_path)
        if scipy.sparse.issparse(slice_i.X):
            slice_i.X = slice_i.X.toarray()
        slice_i.obs = pd.DataFrame({'Ground_Truth': slice_i.obs.iloc[:, 0].values}, index=slice_i.obs.index)
        # n_counts = slice_i.obs['n_genes']
        Ground_Truth = slice_i.obs['Ground_Truth']
        # slice_i.obs = pd.DataFrame({'n_counts': n_counts, 'Ground_Truth': Ground_Truth})
        slice_i.obs = pd.DataFrame({'Ground_Truth': Ground_Truth})
        # slice_i.var = pd.DataFrame({'n_counts': slice_i.var['n_cells']})
        slices.append(slice_i)
    return slices
                         
def align_slices(slices, use_gpu=True):
    alignments = {}
    backend = ot.backend.TorchBackend()
    for i in range(len(slices) - 1):
        pi = pst.pairwise_align(slices[i], slices[i + 1], backend=backend, use_gpu=use_gpu)
        alignments[f'pi{i+1}{i+2}'] = pi
    return alignments

def aligned_slices_with_label(slices):
    alignments = align_slices(slices)
    pis = [alignments[f'pi{i}{i+1}'] for i in range(1, len(slices))]
    new_slices = pst.stack_slices_pairwise(slices, pis)
    return  pis,new_slices

def center_align(slices, use_gpu=True):
    initial_slice = slices[0].copy()
    lmbda = len(slices)*[1/len(slices)]
    pst.filter_for_common_genes(slices)
    b = []
    for i in range(len(slices)):
        b.append(pst.match_spots_using_spatial_heuristic(slices[0].X, slices[i].X))
    center_slice, pis = pst.center_align(initial_slice, slices, lmbda, random_seed = 5, pis_init = b, use_gpu=use_gpu)
    return center_slice,pis

##############################     Metric 1 PAA   ####################################
def create_binary_matrix(slice, n_categories):
    binary_matrix = np.zeros((slice.n_obs, n_categories))
    for idx, cat in enumerate(slice.obs['Ground_Truth'].cat.codes):
        binary_matrix[idx, cat] = 1  
    return binary_matrix

def calculate_PAA(slices, n_categories, backend=ot.backend.TorchBackend(), use_gpu=True):
    total_accuracy = 0
    num_pairs = 0
    for i in range(len(slices)):
        for j in range(i + 1, len(slices)): 
            binary_matrix_i = create_binary_matrix(slices[i], n_categories)
            binary_matrix_j = create_binary_matrix(slices[j], n_categories)
            pi = pst.pairwise_align(slices[i], slices[j], backend=backend, use_gpu=use_gpu)
            matched_pairs = np.dot(binary_matrix_i, binary_matrix_j.T)
            total_accuracy += np.sum(pi * matched_pairs)
            num_pairs += 1
    ave_accuracy = total_accuracy / num_pairs  
    print("PAA:")
    print(ave_accuracy)
    return ave_accuracy


def calculate_center_PAA(center_slice, slices, pis_center, n_categories,backend = ot.backend.TorchBackend(),use_gpu=True):
    total_accuracy=0
    binary_matrix_center = create_binary_matrix(center_slice, n_categories)
    for i in range(len(pis_center)):   
        binary_matrix_i = create_binary_matrix(slices[i], n_categories)
        pi = pis_center[i]
        matched_pairs = np.dot(binary_matrix_center, binary_matrix_i.T)
        total_accuracy += np.sum(pi * matched_pairs)

    ave_accuracy = total_accuracy / (len(slices) * (len(slices) - 1))
    print("The value of ave_accuracy_center is;",ave_accuracy)
    return ave_accuracy
##############################     Metric 1 PAA   ####################################




##############################     Metric 2 SCS   ####################################
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
##############################     Metric 2 SCS   ####################################

##############################     Metric 3 LTARI   ####################################
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

def compute_average_ltari_center(center_slice, slices, k=1):
    reference_slice = center_slice
    ref_coords = reference_slice.obsm['spatial']    
    nn_model = NearestNeighbors(n_neighbors=k).fit(ref_coords)    
    ltari_values=[]
    for query_index in range(len(slices)):
        query_slice = slices[query_index]
        query_coords = query_slice.obsm['spatial']    
        _, nearest_indices = nn_model.kneighbors(query_coords)                
        if k == 1:
            transferred_labels = reference_slice.obs['Ground_Truth'].iloc[nearest_indices.flatten()].values
        else:
            transferred_labels = np.array([reference_slice.obs['Ground_Truth'].iloc[indices].mode()[0] for indices in nearest_indices])
        ari = adjusted_rand_score(query_slice.obs['Ground_Truth'], transferred_labels)
        ltari_values.append(ari)
    LTARI_NN = np.mean(ltari_values)
    print('The value of LTARI_NN_center is:',LTARI_NN)
    return LTARI_NN
##############################     Metric 3 LTARI   ####################################


def whole_process(data_dir,save_dir):
    backend = ot.backend.TorchBackend()
    use_gpu=True
    start = time.time()
    slices = load_slices_h5ad_scc(data_dir)
    _, aligned_slices = aligned_slices_with_label(slices)
    save_dir = os.path.join(save_dir, "paste_pairwise_aligned_slices")
    os.makedirs(save_dir, exist_ok=True)
    for i, slice in enumerate(aligned_slices):
        save_path = os.path.join(save_dir, f"pairwise_aligned_slice_{i}.h5ad")
        sc.write(save_path, slice)

    unique_layers = set()
    for slice in slices:
        unique_layers.update(slice.obs['Ground_Truth'].unique())
    n_categories = len(unique_layers)
    PAA_pairwise = calculate_PAA(slices,n_categories)
    
    _, aligned_slices = aligned_slices_with_label(slices)
    coherence_pairwise = average_spatial_coherence_score(aligned_slices)

    LTARI_NN_pairwise = compute_average_ltari(aligned_slices)

    metrics_data = [
        {"Metric": "PAA", "Value": PAA_pairwise},
        {"Metric": "SCS", "Value": coherence_pairwise},
        {"Metric": "LTARI", "Value": LTARI_NN_pairwise}
    ]

    metrics_df = pd.DataFrame(metrics_data)
    data_dir_name = os.path.basename(os.path.normpath(data_dir))
    csv_filename = f"{data_dir_name}_pairwise_metrics.csv"
    metrics_df.to_csv(os.path.join(save_dir, csv_filename), index=False)


aligned_slices = whole_process(args.data_dir,args.save_dir)

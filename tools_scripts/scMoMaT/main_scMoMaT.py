import time
import h5py
import torch
import scipy
import sys, os
import scmomat
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from util import read_fs_label, h5_to_matrix, read_h5_data

parser = argparse.ArgumentParser("scMoMaT")
parser.add_argument('--path1', metavar='DIR', nargs='+', default=[], help='path to RNA')
parser.add_argument('--path2', metavar='DIR', nargs='+', default=[], help='path to ADT')
parser.add_argument('--path3', metavar='DIR', nargs='+', default=[], help='path to ATAC')
parser.add_argument('--cty_path', metavar='DIR', nargs='+', default=[], help='path to train cty1')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

# The scMoMaT script for vertical/mosaic/cross integration requires multi-batch multi-modal data with bridge modality as input. The output is a joint graph (dimensionality reduction).
# example for vertical integration (RNA+ADT)
# python main_scMoMaT.py --path1 "../../data/dataset_final/D3/rna.h5" --path2 "../../data/dataset_final/D3/adt.h5"  --save_path "../../result/vertical integration/embedding/D3/scMoMaT/"
# example for vertical integration (RNA+ADT+ATAC)
# python main_scMoMaT.py --path1 "../../data/dataset_final/D23/rna.h5" --path2 "../../data/dataset_final/D23/adt.h5" --path3 "../../data/dataset_final/D23/atac.h5"  --save_path "../../result/vertical integration/embedding/D23/scMoMaT/"
# example for cross integration (multiple RNA+ADT)
# python main_scMoMaT.py --path1 "../../data/dataset_final/D51/rna1.h5" "../../data/dataset_final/D51/rna2.h5" --path2 "../../data/dataset_final/D51/adt1.h5"  "../../data/dataset_final/D51/adt2.h5" --save_path "../../result/embedding/cross integration/D51/scMoMaT"
# example for mosaic integration ([RNA, RNA+ADT, ADT])
# python main_scMoMaT.py --path1 "../../data/dataset_final/D38/rna1.h5" "../../data/dataset_final/D38/rna2.h5"  None --path2  None "../../data/dataset_final/D38/adt2.h5" "../../data/dataset_final/D38/adt3.h5"  --path3 None None None --save_path "../../result/embedding/mosaic integration/D38/scMoMaT/"


begin_time = time.time()
def run_scMoMaT(file_paths):
    processed_data,feature_num = read_h5_data(file_paths["rna_path"], file_paths["adt_path"], file_paths["atac_path"], len(file_paths["rna_path"]))
    counts_rnas = processed_data['rna']
    counts_adts = processed_data['adt']
    counts_atacs = processed_data['atac']
    rna_none = all(element is None for element in counts_rnas)
    adt_none = all(element is None for element in counts_adts)
    atac_none = all(element is None for element in counts_atacs)
    if feature_num[0]!="None":
        genes = np.array(["rna" + str(i) for i in range(1, feature_num[0] + 1)])
    if feature_num[1]!="None":
        adt = np.array(["adt" + str(i) for i in range(1, feature_num[1] + 1)])
    if feature_num[2]!="None":
        atac = np.array(["atac" + str(i) for i in range(1, feature_num[2] + 1)])
    if not rna_none and not adt_none and atac_none:
        feats_name = {"rna": genes, "adt": adt}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[0]]), "rna":counts_rnas, "adt": counts_adts}
    elif not rna_none and adt_none and not atac_none:
        feats_name = {"rna": genes, "atac": atac}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[0]]), "rna":counts_rnas, "atac": counts_atacs}
    elif rna_none and not adt_none and not atac_none:
        feats_name = {"adt": adt, "atac": atac}
        print(file_paths[list(file_paths.keys())[1]],"###")
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[1]]), "adt":counts_adts, "atac": counts_atacs}
    elif not rna_none and not adt_none and not atac_none:
        feats_name = {"rna": genes, "adt": adt, "atac": atac}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[0]]), "rna":counts_rnas,"adt":counts_adts, "atac": counts_atacs}
    elif not rna_none and adt_none and atac_none:
        feats_name = {"rna": genes}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[0]]), "rna":counts_rnas}
    elif rna_none and not adt_none and atac_none:
        feats_name = {"adt": adt}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[1]]), "adt":counts_adts}
    elif rna_none and adt_none and not atac_none:
        feats_name = {"atac": atac}
        counts = {"feats_name": feats_name, "nbatches": len(file_paths[list(file_paths.keys())[2]]), "atac":counts_atacs}

    ######### training ############
    K = 30
    lamb = 0.001 
    T = 4000
    interval = 1000
    batch_size = 0.1
    lr = 1e-2
    seed = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(counts['rna'][1].shape)
    #print(counts['rna'][2].shape)
    print(counts['rna'][0].shape)
    #print(counts['adt'][1].shape)
    #print(counts['adt'][2].shape)
    #print(counts['adt'][0].shape)
    #print(counts['adt'][3].shape)
    #print(counts['atac'][1].shape)
    #print(counts['atac'][4].shape)

    # run and get embedding
    model = scmomat.scmomat_model(counts = counts, K = K, batch_size = batch_size, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
    losses = model.train_func(T = T)
    zs = model.extract_cell_factors()
    
    
    # post-processing
    n_neighbors = 100
    r = None
    knn_indices, knn_dists = scmomat.calc_post_graph(zs, n_neighbors, njobs = 8, r = r)
    umap_embedding = scmomat.calc_umap_embedding(knn_indices = knn_indices, knn_dists = knn_dists, n_components = 2, n_neighbors = n_neighbors, min_dist = 0.20, random_state = 0)

    # run and get feature importance score
    #labels_real = read_fs_label(args.cty_path)
    #T = 2000
    #model2 = scmomat.scmomat_retrain(model = model, counts =  counts, labels = labels_real, lamb = lamb, device = device)
    #losses = model2.train(T = T)
    #marker_score = model2.extract_marker_scores()
    return knn_indices, knn_dists, umap_embedding#, marker_score


# run method
file_paths = {
    "rna_path": args.path1,
    "adt_path": args.path2,
    "atac_path": args.path3
}
knn_indices, knn_dists, umap_embedding = run_scMoMaT(file_paths)
end_time = time.time()
all_time = end_time - begin_time
print(all_time)

# save results
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/knn_indices.h5", 'w')
file.create_dataset('data', data=knn_indices)
file.close()
file = h5py.File(args.save_path+"/knn_dists.h5", 'w')
file.create_dataset('data', data=knn_dists)
file.close()
file = h5py.File(args.save_path+"/umap_embedding.h5", 'w')
file.create_dataset('data', data=umap_embedding)
file.close()


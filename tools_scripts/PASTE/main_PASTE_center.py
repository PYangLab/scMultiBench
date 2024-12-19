import os
import ot
import time
import glob
import scipy
import random
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import paste as pst
# %%
parser = argparse.ArgumentParser('PASTE_centre')
parser.add_argument('--data_dir', default='../unified_data/SCC/patient_2/', help='path to the data directory')
parser.add_argument('--save_dir', default='./aligned_slices/', help='path to save the output data')
parser.add_argument('--number', type=int, required=True, help='Number prefix for filenames')
args = parser.parse_args()

# The PASTE_centre script for cross-integration requires spatial data in 'h5ad' format as input, including both gene expression data and spatial coordinates. The output is aligned coordinates (spatial registration).
# run commond for PASTE_centre
# python main_PASTE_centre.py --data_dir '../../data/dataset_final/D60/processed/patient_2/' --save_dir '../../result/registration/D60/PASTE_centre/patient_2'


backend = ot.backend.TorchBackend()
def load_slices_h5ad_scc(data_dir):
    slices = []
    file_paths = glob.glob(data_dir + "*.h5ad")
    for file_path in file_paths:
        slice_i = sc.read_h5ad(file_path)
        if scipy.sparse.issparse(slice_i.X):
            slice_i.X = slice_i.X.toarray()
        slice_i.obs = slice_i.obs[[]]
        slices.append(slice_i)
    return slices
                         
def center_align(slices):
    initial_slice = slices[0].copy()
    lmbda = len(slices)*[1/len(slices)]
    pst.filter_for_common_genes(slices)
    #b = []
    #for i in range(len(slices)):
    #    b.append(pst.match_spots_using_spatial_heuristic(slices[0].X, slices[i].X))
    center_slice, pis = pst.center_align(initial_slice, slices, lmbda, random_seed = 5, backend = ot.backend.TorchBackend(), use_gpu=True) #pis_init = b,
    return center_slice,pis

def whole_process(data_dir,save_dir):
    slices = load_slices_h5ad_scc(data_dir)
    center_slice,pis_center = center_align(slices)
    center, new_slices_center = pst.stack_slices_center(center_slice, slices, pis_center)
    os.makedirs(save_dir, exist_ok=True)
    for i, slice in enumerate(new_slices_center):
        save_path = os.path.join(save_dir, f"aligned_slice_{i}.h5ad")
        sc.write(save_path, slice)
    return new_slices_center
        
aligned_slices = whole_process(args.data_dir,args.save_dir)

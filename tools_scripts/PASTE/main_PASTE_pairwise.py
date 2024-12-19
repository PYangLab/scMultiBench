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
parser = argparse.ArgumentParser('PASTE_pairwise')
parser.add_argument('--data_dir', default='../unified_data/SCC/patient_2/', help='path to the data directory')
parser.add_argument('--save_dir', default='./aligned_slices/', help='path to save the output data')
args = parser.parse_args()

# The PASTE_pairwise script for cross-integration requires spatial data in 'h5ad' format as input, including both gene expression data and spatial coordinates. The output is aligned coordinates (spatial registration).
# run commond for PASTE_pairwise
# python main_PASTE_pairwise.py --data_dir '../../data/dataset_final/D60/processed/patient_2/' --save_dir '../../result/registration/D60/PASTE_pairwise/patient_2'

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

def whole_process(data_dir,save_dir):
    slices = load_slices_h5ad_scc(data_dir)
    _, aligned_slices = aligned_slices_with_label(slices)
    os.makedirs(save_dir, exist_ok=True)
    for i, slice in enumerate(aligned_slices):
        save_path = os.path.join(save_dir, f"aligned_slice_{i}.h5ad")
        sc.write(save_path, slice)
    return aligned_slices

aligned_slices = whole_process(args.data_dir,args.save_dir)

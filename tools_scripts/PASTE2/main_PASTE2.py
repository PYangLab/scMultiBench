# %%
import os
import ot
import math
import glob
import scipy
import random
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
import networkx as nx
from paste2 import PASTE2, projection

parser = argparse.ArgumentParser('paste2')
parser.add_argument('--data_dir', default='../unified_data/SCC/patient_2/', help='path to the data directory')
parser.add_argument('--save_dir', default='./aligned_slices/', help='path to save the output data')
args = parser.parse_args()

# The PASTE2 script for cross-integration requires spatial data in 'h5ad' format as input, including both gene expression data and spatial coordinates. The output is aligned coordinates (spatial registration).
# run commond for PASTE2
# python main_PASTE2.py --data_dir '../../data/dataset_final/D60/processed/patient_2/' --save_dir '../../result/registration/D60/PASTE_centre/patient_2'

# %%
def load_slices_h5ad(data_dir):
    slices = []
    file_paths = glob.glob(data_dir + "*.h5ad")
    for file_path in file_paths:
        slice_i = sc.read_h5ad(file_path)
        if scipy.sparse.issparse(slice_i.X):
            slice_i.X = slice_i.X.toarray()
        slices.append(slice_i)
    return slices

def align_slices(slices):
    # Preprocess each slice
    for slice_i in slices:
        sc.pp.normalize_total(slice_i, target_sum=1e4)
        sc.pp.log1p(slice_i)
        sc.pp.highly_variable_genes(slice_i, n_top_genes=2000, subset=True)
    
    alignments = {}
    for i in range(len(slices) - 1):
        pi = PASTE2.partial_pairwise_align(slices[i], slices[i + 1], s=0.7,dissimilarity = 'pca')
        alignments[f'pi{i+1}{i+2}'] = pi
    return alignments

# %%
def whole_process(data_dir,save_dir):
    slices = load_slices_h5ad(data_dir)
    alignments = align_slices(slices)
    pis = [alignments[f'pi{i}{i+1}'] for i in range(1, len(slices))]
    new_slices = projection.partial_stack_slices_pairwise(slices, pis)
    for i, slice in enumerate(new_slices):
        save_path = os.path.join(save_dir, f"aligned_slice_{i}.h5ad")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sc.write(save_path, slice)
    return new_slices

new_slices = whole_process(args.data_dir,args.save_dir)


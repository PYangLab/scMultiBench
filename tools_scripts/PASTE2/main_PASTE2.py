# %%
import os
import glob
import scipy
import random
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from paste2 import PASTE2, projection

random.seed(1)
parser = argparse.ArgumentParser('paste2')
parser.add_argument('--data_dir', default='../unified_data/SCC/patient_2/', help='path to the data directory')
parser.add_argument('--save_dir', default='./aligned_slices/', help='path to save the output data')
args = parser.parse_args()

# %%
def load_slices_h5ad(data_dir):
    slices = []
    file_paths = glob.glob(data_dir + "*.h5ad")
    for file_path in file_paths:
        slice_i = sc.read_h5ad(file_path)
        
        if scipy.sparse.issparse(slice_i.X):
            slice_i.X = slice_i.X.toarray()
        
        n_counts = slice_i.obs['n_genes']
        layer_guess_reordered = slice_i.obs['Ground_Truth']
        slice_i.obs = pd.DataFrame({'n_counts': n_counts, 'Ground_Truth': layer_guess_reordered})
        
        slice_i.var = pd.DataFrame({'n_counts': slice_i.var['n_cells']})
        slices.append(slice_i)
    
    return slices

# %%
def align_slices(slices):
    alignments = {}
    for i in range(len(slices) - 1):
        pi = PASTE2.partial_pairwise_align(slices[i], slices[i + 1],  s=0.7)
        alignments[f'pi{i+1}{i+2}'] = pi
    return alignments

# %%
def whole_process(data_dir,save_dir):
    slices = load_slices_h5ad(data_dir)
    alignments = align_slices(slices)
    pis = [alignments[f'pi{i}{i+1}'] for i in range(1, len(slices))]
    new_slices = projection.partial_stack_slices_pairwise(slices, pis)
    for i, slice in enumerate(new_slices):
        save_path = os.path.join(save_dir, f"paste2_aligned_slice_{i}.h5ad")
        sc.write(save_path, slice)
    return new_slices
    

# %%
new_slices = whole_process(args.data_dir,args.save_dir)


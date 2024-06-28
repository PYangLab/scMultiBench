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
parser = argparse.ArgumentParser('paste')
parser.add_argument('--data_dir', default='../unified_data/SCC/patient_2/', help='path to the data directory')
parser.add_argument('--save_dir', default='./aligned_slices/', help='path to save the output data')
args = parser.parse_args()


def load_slices_h5ad_scc(data_dir):
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


def center_align(slices):
    initial_slice = slices[0].copy()
    lmbda = len(slices)*[1/len(slices)]
    pst.filter_for_common_genes(slices)
    b = []
    for i in range(len(slices)):
        b.append(pst.match_spots_using_spatial_heuristic(slices[0].X, slices[i].X))
    center_slice, pis = pst.center_align(initial_slice, slices, lmbda, random_seed = 5, pis_init = b)
    return center_slice,pis

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
    center_slice,pis_center = center_align(slices)
    center, new_slices_center = pst.stack_slices_center(center_slice, slices, pis_center)
    save_dir = os.path.join(data_dir, "paste_center_aligned_slices")
    os.makedirs(save_dir, exist_ok=True)
    for i, slice in enumerate(new_slices_center):
        save_path = os.path.join(save_dir, f"center_aligned_slice_{i}.h5ad")
        sc.write(save_path, slice)
    return aligned_slices

aligned_slices = whole_process(args.data_dir,args.save_dir)

# python paste_code.py --data_dir '../unified_data/SCC/patient_2/' --save_dir './aligned_slices/




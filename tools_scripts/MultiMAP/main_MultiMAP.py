import os
import argparse
import scanpy as sc
import anndata
import MultiMAP
import h5py
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix
import time

begin_time = time.time()
parser = argparse.ArgumentParser("MultiMAP")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train gene')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train peak')
parser.add_argument('--path3', metavar='DIR', default='NULL', help='path to train gene activity score')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

def runMultiMAP(rna_path, atac_peaks_path, atac_genes_path): 
    with h5py.File(rna_path, "r") as f:
            X = np.mat(np.array(f['matrix/data']).transpose())
            rna = AnnData(X=X)
            rna.X = csr_matrix(np.matrix(rna.X))

    with h5py.File(atac_peaks_path, "r") as f:
            X = np.mat(np.array(f['matrix/data']).transpose())
            atac_peaks = AnnData(X=X)
            atac_peaks.X = csr_matrix(np.matrix(atac_peaks.X))

    with h5py.File(atac_genes_path, "r") as f:
            X = np.mat(np.array(f['matrix/data']).transpose())
            atac_genes = AnnData(X=X)
            atac_genes.X = csr_matrix(np.matrix(atac_genes.X))
            
    print(rna.X.shape)
    print(atac_peaks.X.shape)
    print(atac_genes.X.shape)
    MultiMAP.TFIDF_LSI(atac_peaks)
    atac_genes.obsm['X_lsi'] = atac_peaks.obsm['X_lsi'].copy()

    rna_pca = rna.copy()
    sc.pp.scale(rna_pca)
    sc.pp.pca(rna_pca)
    rna.obsm['X_pca'] = rna_pca.obsm['X_pca'].copy()
    
    adata = MultiMAP.Integration([rna, atac_genes], ['X_pca', 'X_lsi'], strengths=[0,1])
    result = adata.obsm['X_multimap']
    return result


result = runMultiMAP(args.path1, args.path2, args.path3) # 2is peak, 3 is gas

end_time = time.time()
all_time = end_time - begin_time
print(all_time)


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")


file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")

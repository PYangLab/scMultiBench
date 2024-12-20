import os
import time
import h5py
import anndata
import argparse
import MultiMAP
import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

begin_time = time.time()
parser = argparse.ArgumentParser("MultiMAP")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to train gene')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to train peak')
parser.add_argument('--path3', metavar='DIR', default='NULL', help='path to train gene activity score')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

# The MultiMAP script for diagonal integration requires RNA and ATAC data as input, where ATAC needs to be transformed into gene activity score. The output is a joint embedding (dimensionality reduction).
# run commond for MultiMAP
# python main_MultiMAP.py --path1 "../../data/dataset_final/D27/rna.h5" --path2 "../../data/dataset_final/D27/atac_peak.h5" --path3 "../../data/dataset_final/D27/atac_gas.h5" --save_path "../../result/embedding/diagonal integration/D27/MultiMAP/"

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


import os
import time
import h5py
import torch
import random
import anndata
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler
from scMVP.inference import MultiPosterior, MultiTrainer
from sklearn.feature_extraction.text import TfidfTransformer
from scMVP.dataset import LoadData,GeneExpressionDataset, CellMeasurement
from scMVP.models import VAE_Attention, Multi_VAE_Attention, VAE_Peak_SelfAttention
    
torch.set_num_threads(40)
parser = argparse.ArgumentParser("scMVP")
parser.add_argument('--path1', metavar='DIR', default='NULL', help='path to RNA')
parser.add_argument('--path2', metavar='DIR', default='NULL', help='path to ATAC')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

# The scMVP script for vertical integration requires RNA and ATAC data as input. The output is a joint embedding (dimensionality reduction).
# run commond for scMVP
# python main_scMVP.py --path1 "../../data/dataset_final/D15/rna.h5" --path2 "../../data/dataset_final/D15/atac.h5" --save_path "../../result/embedding/vertical integration/D15/scMVP/"

def run_MVP(rna_path, atac_path):
    with h5py.File(rna_path, "r") as f:
        rna_X = np.array(f['matrix/data']).transpose()
        # pre-processing for rna data
        adata= anndata.AnnData(X=rna_X)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        selected_features = adata.var['highly_variable'].values
        adata = adata[:,selected_features]
        rna_X = adata.X
        rna_X = rna_X.transpose()

        rna_barcode = np.array(f['matrix/barcodes']).transpose()
        rna_feature = np.array(f['matrix/features']).transpose()[selected_features]

    with h5py.File(atac_path, "r") as f:
        atac_X = np.asarray(np.mat(np.array(f['matrix/data']))).transpose()
        # pre-processing for atac data
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(atac_X)
        atac_X = tfidf.todense()
        scaler = StandardScaler()
        atac_X = scaler.fit_transform(atac_X).transpose()

        atac_barcode = np.array(f['matrix/barcodes']).transpose()
        atac_feature = np.array(f['matrix/features']).transpose()

    sparse_rna = coo_matrix(rna_X)
    sparse_atac = coo_matrix(atac_X)
    
    with open(rna_path.replace('.h5', '.mtx'), 'w') as f:
        f.write('%%MatrixMarket matrix coordinate integer general\n')
        f.write('{} {} {}\n'.format(sparse_rna.shape[0], sparse_rna.shape[1], sparse_rna.nnz))
        for i, j, v in zip(sparse_rna.row + 1, sparse_rna.col + 1, sparse_rna.data):
            f.write('{} {} {}\n'.format(i, j, int(v)))
            
    with open(atac_path.replace('.h5', '.mtx'), 'w') as f:
        f.write('%%MatrixMarket matrix coordinate integer general\n')
        f.write('{} {} {}\n'.format(sparse_atac.shape[0], sparse_atac.shape[1], sparse_atac.nnz))
        for i, j, v in zip(sparse_atac.row + 1, sparse_atac.col + 1, sparse_atac.data):
            f.write('{} {} {}\n'.format(i, j, int(v)))

    dir_path = os.path.dirname(rna_path)
    file_name = "barcode.txt"
    txt_path = os.path.join(dir_path, file_name)
    np.savetxt(txt_path, rna_barcode.transpose(), fmt='%s', delimiter='\n')
    dir_path = os.path.dirname(rna_path)
    file_name = "rna_feature.txt"
    txt_path = os.path.join(dir_path, file_name)
    np.savetxt(txt_path, rna_feature.transpose(), fmt='%s', delimiter='\n')
    dir_path = os.path.dirname(rna_path)
    file_name = "atac_feature.txt"
    txt_path = os.path.join(dir_path, file_name)
    np.savetxt(txt_path, atac_feature.transpose(), fmt='%s', delimiter='\n')
    
    input_path = dir_path+"/"
    sciCAR_cellline_dataset = {
                    "gene_names": 'rna_feature.txt',
                    "gene_expression": os.path.basename(rna_path).replace('.h5', '.mtx'),
                    "gene_barcodes": 'barcode.txt',
                    "atac_names": 'atac_feature.txt',
                    "atac_expression": os.path.basename(atac_path).replace('.h5', '.mtx'),
                    "atac_barcodes": 'barcode.txt'
                    }

    dataset = LoadData(dataset=sciCAR_cellline_dataset,data_path=input_path,
                           dense=False,gzipped=False, atac_threshold=0.001,
                           cell_threshold=1)

    n_epochs = 10
    lr = 1e-3
    use_batches = False
    use_cuda = True # False if using CPU
    n_centroids = 5 
    n_alfa = 1.0
    multi_vae = Multi_VAE_Attention(dataset.nb_genes, len(dataset.atac_names), n_batch=0, n_latent=20, n_centroids=n_centroids, n_alfa = n_alfa, mode="mm-vae") # should provide ATAC num, alfa, mode and loss type
    trainer = MultiTrainer(
        multi_vae,
        dataset,
        train_size=0.9,
        use_cuda=use_cuda,
        frequency=5,
    )
    trainer.train(n_epochs=n_epochs, lr=lr)
    # create posterior from trained model
    full = trainer.create_posterior(trainer.model, dataset, indices=np.arange(len(dataset)),type_class=MultiPosterior)
    latent, latent_rna, latent_atac, cluster_gamma, cluster_index, batch_indices, labels = full.sequential().get_latent()
    return latent

begin_time = time.time()
result = run_MVP(args.path1, args.path2)
result = np.transpose(result)
end_time = time.time()
all_time = end_time - begin_time
print(all_time)
print(result.shape)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")

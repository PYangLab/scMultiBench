import os
import time
import h5py
import muon
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import multigrate as mtg
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser("Multigrate")
parser.add_argument('--path1', metavar='DIR', default="", help='path to train RNA')
parser.add_argument('--path2', metavar='DIR', default="", help='path to train RNA of CITEseq')
parser.add_argument('--path3', metavar='DIR', default="", help='path to train ADT of CITEseq')
parser.add_argument('--path4', metavar='DIR', default="", help='path to train ADT')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
parser.add_argument('--epochs', type = int, default=200, help='')
parser.add_argument('--lr', type = float, default=1e-4, help='')
args = parser.parse_args()

# This Multigrate script is designed for mosaic integration for [RNA, RNA+ADT, ADT] data type.
# example for mosaic integration ([RNA, RNA+ADT, ADT])
# python main_Multigrate_mosaic_rna_adt.py --path1  "../../data/dataset_final/D39/rna1.h5" --path2 "../../data/dataset_final/D39/rna2.h5" --path3 "../../data/dataset_final/D39/adt2.h5" --path4 "../../data/dataset_final/D39/adt3.h5" --save_path "../../result/embedding/mosaic integration/D39/Multigrate/"

begin_time = time.time()
def process_rna(adata_rna,barcodes,featurs):
    adata_rna.layers['counts'] = adata_rna.X
    adata_rna.obs.index = barcodes
    adata_rna.var_names = featurs
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    return adata_rna
    
def process_atac(adata_atac,barcodes,featurs):
    adata_atac.layers['counts'] = adata_atac.X
    adata_atac.obs.index = barcodes
    adata_atac.var_names = featurs
    sc.pp.normalize_total(adata_atac, target_sum=1e4)
    sc.pp.log1p(adata_atac)
    return adata_atac

def process_adt(adata_adt,barcodes,featurs):
    adata_adt.layers['counts'] = adata_adt.X
    adata_adt.obs.index = barcodes
    adata_adt.var_names = featurs
    muon.prot.pp.clr(adata_adt)
    return adata_adt
    
def h5_to_matrix(path):
    with h5py.File(path, "r") as f:
        X = csr_matrix(np.mat(np.array(f['matrix/data']).transpose()))
        barcodes = []
        for key in f['matrix/barcodes']:
            barcodes.append(key.decode('UTF-8'))
        features = []
        for key in f['matrix/features']:
            features.append(key.decode('UTF-8'))
    return X, barcodes, features
    
rna_path1 = args.path1
rna_path2 = args.path2
adt_path2 = args.path3
adt_path3 = args.path4

rna1, barcodes1, rna_features = h5_to_matrix(rna_path1)
rna2, barcodes2, rna_features = h5_to_matrix(rna_path2)
adt2, barcodes2, adt_features = h5_to_matrix(adt_path2)
adt3, barcodes3, adt_features = h5_to_matrix(adt_path3)

adata_rna1 = process_rna(ad.AnnData(rna1),barcodes1, rna_features)
adata_rna2 = process_rna(ad.AnnData(rna2),barcodes2, rna_features)
adata_adt2 = process_adt(ad.AnnData(adt2), barcodes2, adt_features)
adata_adt3 = process_adt(ad.AnnData(adt3), barcodes3, adt_features)

adata_rna1.obs['Modality'] = "rna"
adata_rna2.obs['Modality'] = "cite"
adata_adt2.obs['Modality'] = "cite"
adata_adt3.obs['Modality'] = "adt"

adata_rna = ad.concat([adata_rna1, adata_rna2], merge='same')
sc.pp.highly_variable_genes(adata_rna, n_top_genes=4000,  batch_key='Modality')
adata_rna1 = adata_rna1[:, adata_rna.var.highly_variable]
adata_rna2 = adata_rna2[:, adata_rna.var.highly_variable]

adata_rna1.obs.index = "rna1-" +adata_rna1.obs.index
adata_rna2.obs.index = "cite-" +adata_rna2.obs.index
adata_adt2.obs.index = "cite-" +adata_adt2.obs.index
adata_adt3.obs.index = "adt3-" +adata_adt3.obs.index
    
adata = mtg.data.organize_multiome_anndatas(
    adatas = [[adata_rna1, adata_rna2, None], [None, adata_adt2, adata_adt3]],
    layers = [['counts', 'counts', None], [None, None, None]],
)
adata

mtg.model.MultiVAE.setup_anndata(
    adata,
    categorical_covariate_keys=['Modality'],# 'Samplename'],
    rna_indices_end=4000
)

model = mtg.model.MultiVAE(
    adata,
    losses=['nb', 'mse'],
    loss_coefs={'kl': 1e-3,
               'integ': 4000,
               },
    integrate_on='Modality',
    mmd='marginal',
)
model.train( lr=args.lr)
model.get_latent_representation()
result = adata.obsm['latent']
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

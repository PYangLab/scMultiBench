import os
import h5py
import time
import random
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from anndata import AnnData, read_h5ad
from sciPENN.sciPENN_API import sciPENN_API
from sciPENN.Preprocessing import preprocess


parser = argparse.ArgumentParser("sciPENN")
parser.add_argument('--path1', metavar='DIR', nargs='+', default="", help='path to RNA')
parser.add_argument('--path2', metavar='DIR', nargs='+', default="", help='path to ADT')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
parser.add_argument('--seed',  type=int,  default=1, help='path to save the output data')
args = parser.parse_args()

# The sciPENN script for vertical/cross integration requires one/multiple matched RNA+ADT data as input. The output is a joint embedding (dimensionality reduction).
# run commond for sciPENN (RNA+ADT)
#python main_sciPENN.py --path1 "../../data/dataset_final/SD1/rna.h5" --path2 "../../data/dataset_final/SD1/adt.h5"  --save_path "../../result/embedding/vertical integration/SD1/sciPENN/"
# run commond for sciPENN (multiple RNA+ADT)
#python main_sciPENN.py --path1 "../../data/dataset_final/SD15/rna1.h5" "../../data/dataset_final/SD15/rna2.h5" --path2 "../../data/dataset_final/SD15/adt1.h5"  "../../data/dataset_final/SD15/adt2.h5" --save_path "../../result/embedding/cross integration/SD15/sciPENN"

random.seed(args.seed)
begin_time = time.time()
def load_data_vertical(rna_path,adt_path,i):
  with h5py.File(adt_path, 'r') as f:
    data_adt = np.array(f['matrix/data'])
    barcodes_adt = np.array(f['matrix/barcodes'])
    features_adt = np.array(f['matrix/features'])
  with h5py.File(rna_path, 'r') as f:
    data_rna = np.array(f['matrix/data'])
    barcodes_rna = np.array(f['matrix/barcodes'])
    features_rna = np.array(f['matrix/features'])
    RNA_data = sc.AnnData(X=data_rna.T, obs=pd.DataFrame(index=barcodes_rna), var=pd.DataFrame(index=features_rna))
    ADT_data = sc.AnnData(X=data_adt.T, obs=pd.DataFrame(index=barcodes_adt), var=pd.DataFrame(index=features_adt))
    ADT_data.obs['batch'] = i
    RNA_data.obs['batch'] = i
    
    print(RNA_data.obs)
    return RNA_data, ADT_data
    
def generate_data(RNA_data, ADT_data):
    adata_gene_train = RNA_data.copy()
    adata_protein_train = ADT_data.copy()
    
    # Added by Sichang to make sure the type of data is correct for following codes.
    for ad in [adata_gene_train, adata_protein_train]:
        for key in ad.obs.columns:
            if ad.obs[key].dtype == 'O':
                ad.obs[key] = ad.obs[key].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

        for key in ad.var.columns:
            if ad.var[key].dtype == 'O':
                ad.var[key] = ad.var[key].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return adata_gene_train,adata_protein_train
    
def embedding_output(adata_gene_train, adata_protein_train):

    for adata in adata_gene_train + adata_protein_train:
        adata.obs.index = adata.obs.index.map(str)
        
    adata_gene_combined = adata_gene_train 
    adata_protein_combined = adata_protein_train 
    batch = []
    for i in range(len(adata_gene_combined)):
        adata_gene_combined[i].obs['batch'] = str(adata_gene_combined[i].obs['batch'])
        adata_protein_combined[i].obs['batch'] = str(adata_protein_combined[i].obs['batch'])
        batch.append('batch')
        
    sciPENN = sciPENN_API(gene_trainsets = adata_gene_combined, protein_trainsets = adata_protein_combined, train_batchkeys = batch, min_cells=0, min_genes=0) #min_cells=0, min_genes=0 is added by myself.
    sciPENN.train(quantiles = [0.1, 0.25, 0.75, 0.9], n_epochs = 10000, ES_max = 12, decay_max = 6,
             decay_step = 0.1, lr = 10**(-3), weights_dir = "pbmc_to_pbmc", load = False)
    embedding_output = sciPENN.embed()
    return embedding_output
    

    
def whole_process(rna_paths,adt_paths):
    adata_gene_train_list = []
    adata_protein_train_list = []
    i = 0
    for rna_path, adt_path in zip(rna_paths, adt_paths):
            RNA_data, ADT_data = load_data_vertical(rna_path,adt_path, i)
            adata_gene_train,adata_protein_train = generate_data(RNA_data, ADT_data)
            adata_gene_train_list.append(adata_gene_train)
            adata_protein_train_list.append(adata_protein_train)
            i = i + 1
    standard_var_names = adata_gene_train_list[0].var_names

    for adata in adata_gene_train_list:
        adata.var_names = standard_var_names
    embedding = embedding_output(adata_gene_train_list, adata_protein_train_list)
    return embedding
    
# RUN METHOD
result = whole_process(args.path1, args.path2)
end_time = time.time()
all_time = end_time - begin_time
print(all_time)
print(result.shape)

# SAVE RESULTS
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result.X)
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")

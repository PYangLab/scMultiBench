# %%
import os
import scvi
import h5py
import torch
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scrublet as scr
from scvi.model import TOTALVI

parser = argparse.ArgumentParser('totalVI')
parser.add_argument('--dataset_dir', default='../test_bench/cross/dataset51/', help='path to the data directory')
parser.add_argument('--save_dir', default='./processed_data/', help='path to save the output data')
parser.add_argument('--holdout_batch', default='batch2', help='holdout batch for imputation')
parser.add_argument('--transform_batch',default = 'batch1', help= 'imputation reference')
args = parser.parse_args()

# %%
def load_data_with_batch_and_label(adt_path, rna_path, cty_path, batch_name, adt_features=None):
    with h5py.File(rna_path, 'r') as f:
        data_rna = np.array(f['matrix/data']).T
        barcodes_rna = np.array(f['matrix/barcodes'])

    if adt_path is not None:
        with h5py.File(adt_path, 'r') as f:
            data_adt = np.array(f['matrix/data']).T
    else:
        data_adt = np.zeros((len(barcodes_rna), adt_features))

    cty_df = pd.read_csv(cty_path, skiprows=1, header=None)
    cell_types = cty_df.iloc[:, 1].values

    adata = sc.AnnData(X=data_rna, obs=pd.DataFrame(index=barcodes_rna))
    adata.obs['batch'] = batch_name
    adata.obs['cell_type'] = cell_types
    adata.obsm['protein_expression'] = pd.DataFrame(data_adt, index=adata.obs_names)

    return adata


def sort_key(file_path):
    if file_path is None:
        return float('inf')
    file_name = os.path.basename(file_path)
    batch_num = int(file_name.split(".")[0][3:])
    return batch_num


def get_adata_mod(adt_files, rna_files, cty_files, batch_names):
    adata_list = []
    adt_features = None

    for adt, rna, cty, batch in zip(adt_files, rna_files, cty_files, batch_names):
        if adt is not None and adt_features is None:
            with h5py.File(adt, 'r') as f:
                adt_features = np.array(f['matrix/features']).shape[0]

        adata = load_data_with_batch_and_label(adt, rna, cty, batch, adt_features)
        adata_list.append(adata)
    adata_combined = sc.concat(adata_list, axis=0, join='outer')
    n_total_sample = adata_combined.shape[0]
    return adata_combined, n_total_sample

# %%
def load_files(dataset_dir):
    file_names = os.listdir(dataset_dir)
    adt_files = []
    rna_files = []
    cty_files = []
    batch_names = []

    for file_name in file_names:
        if file_name.startswith("adt") and file_name.endswith(".h5"):
            adt_files.append(os.path.join(dataset_dir, file_name))

        elif file_name.startswith("rna") and file_name.endswith(".h5"):
            rna_files.append(os.path.join(dataset_dir, file_name))
        elif file_name.startswith("cty") and file_name.endswith(".csv"):
            cty_files.append(os.path.join(dataset_dir, file_name))

    adt_files.sort(key=sort_key)
    rna_files.sort(key=sort_key)
    cty_files.sort(key=sort_key)
    batch_names = ["batch" + str(i+1) for i in range(len(rna_files))]

    adt_files_dict = {int(os.path.basename(adt_file).split(".")[0][3:]): adt_file for adt_file in adt_files}
    adt_files_sorted = []
    for i in range(len(rna_files)):
        rna_file_name = os.path.basename(rna_files[i])
        batch_num = int(rna_file_name.split(".")[0][3:])
        
        adt_file = adt_files_dict.get(batch_num)
        
        adt_files_sorted.append(adt_file)

    adata, n_total_sample = get_adata_mod(adt_files_sorted, rna_files, cty_files, batch_names)
    batch = adata.obs.batch.values.ravel()
    return adata,batch

# %%
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

sns.set_theme()
torch.set_float32_matmul_precision("high")

holdout_batch = args.holdout_batch
save_dir = args.save_dir
dataset_dir = args.dataset_dir
transform_batch = args.transform_batch

adata,batch = load_files(dataset_dir)
held_out_proteins = adata.obsm["protein_expression"][batch == holdout_batch].copy()
df = pd.DataFrame(adata.obsm["protein_expression"], index=adata.obs_names)
df.loc[batch == holdout_batch] = np.zeros_like(adata.obsm["protein_expression"][batch == holdout_batch])
adata.obsm["protein_expression"] = df.to_numpy()



sc.pp.highly_variable_genes(
    adata, batch_key="batch", flavor="seurat_v3", n_top_genes=4000, subset=True
)


scvi.model.TOTALVI.setup_anndata(
    adata, batch_key="batch", protein_expression_obsm_key="protein_expression"
)
model = scvi.model.TOTALVI(adata, latent_distribution="normal", n_layers_decoder=2)
model.train()

embedding = model.get_latent_representation()

rna, protein = model.get_normalized_expression(
    n_samples=25, return_mean=True, transform_batch=transform_batch
)

adata.obsm['protein_expression'] = protein.to_numpy()
data_imputed = protein[batch ==holdout_batch]

save_dir = './processed_data/'
os.makedirs(save_dir, exist_ok=True)

imputed_data_filename = "imputed_data.h5"
imputed_data_path = os.path.join(save_dir, imputed_data_filename)
with h5py.File(imputed_data_path, 'w') as file:
    if held_out_proteins is not None:
        file.create_dataset("data", data=data_imputed)
embedding_data_filename = "embeddings.h5"
embeddings_path = os.path.join(save_dir, embedding_data_filename)
with h5py.File(embeddings_path, 'w') as file:
    file.create_dataset("data", data=embedding)





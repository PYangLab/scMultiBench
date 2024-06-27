# title
import os
import time
import h5py
import random
import scglue
import anndata
import argparse
import itertools
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import networkx as nx
from scipy import sparse
from itertools import chain
from anndata import AnnData

random.seed(1)
scglue.plot.set_publication_params()
begin_time = time.time()
parser = argparse.ArgumentParser("GLUE")
parser.add_argument('--path1', metavar='DIR', nargs ="*", default=[], help='path to train gene')
parser.add_argument('--path2', metavar='DIR', nargs ="*", default=[], help='path to train peak')
parser.add_argument('--species', metavar='str', default='Mouse', help='Human or Mouse')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

# read data
## read rna data (raw raw count)
def load_rna(rna_path):
    with h5py.File(rna_path, "r") as f:
        X = np.mat(np.array(f['matrix/data']).transpose())
        rna = AnnData(X=X)
        rna.X = sparse.csr_matrix(np.matrix(rna.X))
        rna.var_names = np.array(f['matrix/features']).astype(str)
        rna.obs_names = np.array(f['matrix/barcodes']).astype(str)
    return(rna)

def convert_gene_name(rna):
    converted_gene_names = []
    for name in rna.var_names:
        if "-" in name:
            parts = name.split("-")
            converted_parts = [part.capitalize() for part in parts]
            converted_name = "-".join(converted_parts)
        else:
            converted_name = name.capitalize()

        if 'rik' in converted_name:
            converted_name = converted_name.upper()
            converted_name = converted_name.replace('RIK', 'Rik')  
        converted_gene_names.append(converted_name)
    rna.var_names = converted_gene_names
    return(rna)

## read atac data (peak raw count)
def load_atac(atac_peaks_path):
    with h5py.File(atac_peaks_path, "r") as f:
        X = np.mat(np.array(f['matrix/data']).transpose())
        atac = AnnData(X=X)
        atac.X = sparse.csr_matrix(np.matrix(atac.X))
        atac.var_names = np.array(f['matrix/features']).astype(str)
        atac.obs_names = np.array(f['matrix/barcodes']).astype(str)
    return(atac)

def preprocess_rna(rna):
    rna.layers["counts"] = rna.X.copy()
    sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=100, svd_solver="auto")
    return (rna)

def preprocess_atac(atac):
    scglue.data.lsi(atac, n_components=100, n_iter=15)
    return (atac)

def runGLUE(rna_path_list, atac_peaks_path_list, species="Mouse"): # claim to have dna as well, missing in its tutorial
    # read data
    ## read rna data (raw raw count)
    rna_list =[]
    for rna_path in rna_path_list:
        rna = load_rna(rna_path)
        rna_list.append(rna)
    rna_list = anndata.concat(rna_list)
    rna = rna_list
    batch = np.concatenate([np.ones(ds.shape[0], dtype=int)*i for i, ds in enumerate(rna_list)])
    rna.obs["batch"] = batch

    ## read atac data (peak raw count)
    atac_list = []
    for atac_peaks_path in atac_peaks_path_list:
        atac = load_atac(atac_peaks_path) 
        atac_list.append(atac)
    atac_list = anndata.concat(atac_list)
    atac = atac_list
    batch = np.concatenate([i * np.ones(ds.shape[0], dtype=int) for i, ds in enumerate(atac_list)])
    atac.obs["batch"] = batch
    
    # graph construction
    #rna.var.head()
    if species == "Human":
        print("Human!!!!!!!!!")
        scglue.data.get_gene_annotation(
            rna, gtf="gencode.v43.chr_patch_hapl_scaff.annotation.gtf.gz",
            gtf_by="gene_name"
        )
        rna = preprocess_rna(rna) 
        atac = preprocess_atac(atac)
        print(rna.shape)
        print(atac.shape)
        
        print(atac.var_names)
        split = atac.var_names.str.split(r"[:-]")
        atac.var["chrom"] = split.map(lambda x: x[0])
        atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
        atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
        atac.var.head()

        # Find all the indices that are duplicated
        duplicates = rna.var.index[rna.var.index.duplicated(keep=False)]
        print("Initial duplicates:", duplicates)

        # Create a new index for the DataFrame
        new_index = rna.var.index.to_list()

        # Iterate over each duplicate and rename accordingly
        for item in duplicates.unique():
            # Find the positions of the duplicates for the current item
            dup_positions = [i for i, x in enumerate(new_index) if x == item]
            # Now iterate over those positions and rename accordingly
            for j, pos in enumerate(dup_positions):
                if j > 0:  # Keep the first occurrence without a suffix
                    new_index[pos] = f"{item}-{j+1}"

        # Assign the new index to the DataFrame
        rna.var.index = pd.Index(new_index)

        # Check if there are any duplicates remaining
        duplicates = rna.var.index[rna.var.index.duplicated(keep=False)]
        print("Remaining duplicates:", duplicates)
        
        rna = rna[:,~rna.var["chrom"].isna()]
        guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
        scglue.graph.check_graph(guidance, [rna, atac])
        
        if np.max(rna.obs["batch"])>0:
            scglue.models.configure_dataset(
                rna, "NB", use_highly_variable=True,
                use_layer="counts", use_rep="X_pca",
                use_batch = "batch"
            )
        else:
            scglue.models.configure_dataset(
                rna, "NB", use_highly_variable=True,
                use_layer="counts", use_rep="X_pca"
            )
            
        if np.max(atac.obs["batch"])>0:
            scglue.models.configure_dataset(
                atac, "NB", use_highly_variable=True,
                use_rep="X_lsi",use_batch = "batch"
            )
        else:
            scglue.models.configure_dataset(
                atac, "NB", use_highly_variable=True,
                use_rep="X_lsi"
            )
            
        guidance_hvf = guidance.subgraph(chain(
            rna.var.query("highly_variable").index,
            atac.var.query("highly_variable").index
        )).copy()
        glue = scglue.models.fit_SCGLUE(
            {"rna": rna, "atac": atac}, guidance_hvf,
            fit_kws={"directory": "glue"}
        )

        
    elif species == "Mouse":
        print("Mouse!!!!!!!!!")
        rna = preprocess_rna(convert_gene_name(rna))
        atac = preprocess_atac(atac)
        scglue.data.get_gene_annotation(
            rna, gtf="gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
            gtf_by="gene_name"
        )
        print(atac.var_names.str)
        split = atac.var_names.str.split(r"[__]")
        atac.var["chrom"] = split.map(lambda x: x[0])
        atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
        atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
        atac.var.head()


        # Find all the indices that are duplicated
        duplicates = rna.var.index[rna.var.index.duplicated(keep=False)]
        print("Initial duplicates:", duplicates)

        # Create a new index for the DataFrame
        new_index = rna.var.index.to_list()

        # Iterate over each duplicate and rename accordingly
        for item in duplicates.unique():
            # Find the positions of the duplicates for the current item
            dup_positions = [i for i, x in enumerate(new_index) if x == item]
            # Now iterate over those positions and rename accordingly
            for j, pos in enumerate(dup_positions):
                if j > 0:  # Keep the first occurrence without a suffix
                    new_index[pos] = f"{item}-{j+1}"

        # Assign the new index to the DataFrame
        rna.var.index = pd.Index(new_index)

        # Check if there are any duplicates remaining
        duplicates = rna.var.index[rna.var.index.duplicated(keep=False)]
        print("Remaining duplicates:", duplicates)

        rna = rna[:,~rna.var["chrom"].isna()]
        rna
        print(rna.var.index.unique())
        print(atac.var.index.unique())

        guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
        scglue.graph.check_graph(guidance, [rna, atac])
        scglue.models.configure_dataset(
            rna, "NB", use_highly_variable=True,
            use_layer="counts", use_rep="X_pca",
            #use_batch = "batch"
        )
        scglue.models.configure_dataset(
            atac, "NB", use_highly_variable=True,
            use_rep="X_lsi",#use_batch = "batch"
        )
        guidance_hvf = guidance.subgraph(chain(
            rna.var.query("highly_variable").index,
            atac.var.query("highly_variable").index
        )).copy()
        glue = scglue.models.fit_SCGLUE(
            {"rna": rna, "atac": atac}, guidance_hvf,
            fit_kws={"directory": "glue"}
        )
    
    else:
        print("Error, please give a species")
        
    # works totally fine
    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)
    result = np.concatenate([rna.obsm["X_glue"],atac.obsm["X_glue"]],0)
    return result
    
result = runGLUE(args.path1, args.path2, args.species) 
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

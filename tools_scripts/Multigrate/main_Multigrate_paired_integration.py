import os
import time
import h5py
import muon
import random
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import multigrate as mtg
import scipy.sparse as sp

parser = argparse.ArgumentParser("Multigrate")
parser.add_argument('--path1', metavar='DIR', nargs='+', default=[], help='path to rna')
parser.add_argument('--path2', metavar='DIR', nargs='+', default=[], help='path to adt')
parser.add_argument('--path3', metavar='DIR', nargs='+', default=[], help='path to atac')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
parser.add_argument('--epochs', type = int, default=200, help='')
parser.add_argument('--bs', type = int, default=256, help='')
parser.add_argument('--lr', type = float, default=1e-3, help='')
args = parser.parse_args()

# The Multigrate script is designed for both vertical and cross integration.
# example for vertical integration (rna+adt)
# python main_Multigrate_paired_integration.py --path1 "../../data/dataset_final/D3/rna.h5" --path2 "../../data/dataset_final/D3/adt.h5"  --save_path "../../result/embedding/vertical integration/D3/Multigrate"
# example for vertical integration (rna+adt+atac)
# python main_Multigrate_paired_integration.py --path1 "../../data/dataset_final/D23/rna.h5" --path2 "../../data/dataset_final/D23/adt.h5"   --path3 "../../data/dataset_final/D23/atac.h5"  --save_path "../../result/embedding/vertical integration/D23/Multigrate"
# example for cross integration (multiple rna+adt)
# python main_Multigrate_paired_integration.py --path1 "../../data/dataset_final/D51/rna1.h5" "../../data/dataset_final/D51/rna2.h5" --path2 "../../data/dataset_final/D51/adt1.h5"  "../../data/dataset_final/D51/adt2.h5" --save_path "../../result/embedding/cross integration/D51/Multigrate"

begin_time = time.time()

def process_rna(adata_rna):
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    print(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, n_top_genes=4000) 
    return adata_rna
    
def process_atac(adata_atac):
    print(adata_atac)
    sc.pp.normalize_total(adata_atac, target_sum=1e4)
    sc.pp.log1p(adata_atac)
    sc.pp.highly_variable_genes(adata_atac, n_top_genes=20000) 
    return adata_atac

def process_adt(adata_adt):
    muon.prot.pp.clr(adata_adt)
    return adata_adt
    
def h5_to_matrix(path):
    with h5py.File(path, "r") as f:
        X = (np.array(f['matrix/data']).transpose())
    return X

def read_h5_data(rna_path=None, adt_path=None, atac_path=None, list_len=3):
    rna_list = []
    adt_list = []
    atac_list = []
    
    # read rna
    if rna_path is not None:
        for i in range(len(rna_path)):
            if rna_path[i] is None:
                rna_list.append(None)
            else:
                rna_list.append(h5_to_matrix(rna_path[i]))
        
    # read adt
    if adt_path is not None:
        for i in range(len(adt_path)):
            if adt_path[i] is None:
                adt_list.append(None)
            else:
                adt_list.append(h5_to_matrix(adt_path[i]))
    
    # read atac
    if atac_path is not None:
        for i in range(len(atac_path)):
            if atac_path[i] is None:
                atac_list.append(None)
            else:
                atac_list.append(h5_to_matrix(atac_path[i]))
                
    print(adt_list)
    print(atac_list)
    
    batch = []
    if rna_path is not None:
        for i in range(len(rna_list)):
            batch.append(np.ones((rna_list[i].shape[0], 1))  + i)
    elif adt_path is not None:
        for i in range(len(adt_list)):
            batch.append(np.ones((adt_list[i].shape[0], 1))  + i)
    elif atac_path is not None:
        for i in range(len(atac_list)):
            batch.append(np.ones((atac_list[i].shape[0], 1))  + i)
        
    if (rna_path is not None) and (adt_path is not None) and (atac_path is not None):
        adata_rna = process_rna(ad.AnnData(np.concatenate(rna_list, axis=0)))
        adata_adt = process_adt(ad.AnnData(np.concatenate(adt_list, axis=0)))
        adata_atac = process_atac(ad.AnnData(np.concatenate(atac_list, axis=0)))
        result = {"rna": adata_rna, "adt": adata_adt, "atac": adata_atac}
        
    if (rna_path is not None) and (adt_path is not None) and (atac_path is None):
        adata_rna = process_rna(ad.AnnData(np.concatenate(rna_list, axis=0)))
        adata_adt = process_adt(ad.AnnData(np.concatenate(adt_list, axis=0)))
        result = {"rna": adata_rna, "adt": adata_adt}
        
    if (rna_path is not None) and (adt_path is None) and (atac_path is not None):
        adata_rna = process_rna(ad.AnnData(np.concatenate(rna_list, axis=0)))
        adata_atac = process_atac(ad.AnnData(np.concatenate(atac_list, axis=0)))
        result = {"rna": adata_rna, "atac": adata_atac}
        
    if (rna_path is None) and (adt_path is not None) and (atac_path is not None):
        adata_adt = process_adt(ad.AnnData(np.concatenate(adt_list, axis=0)))
        adata_atac = process_atac(ad.AnnData(np.concatenate(atac_list, axis=0)))
        result = {"adt": adata_adt, "atac": adata_atac}        
    return result, np.concatenate(batch,0)

def run_Multigrate(file_paths):
    if (file_paths['rna_path'] is not None) and (file_paths['adt_path'] is not None) and (file_paths['atac_path'] is None):
        result, batch = read_h5_data(file_paths["rna_path"],file_paths["adt_path"],file_paths['atac_path'])
        result_list = [] 
        for i in range(len(list(result))):
            temp = []
            temp.append(result[list(result.keys())[i]])
            result_list.append(temp)
        adata = mtg.data.organize_multiome_anndatas(
            adatas = result_list, 
        )
        adata.obs["Batch"] = batch
        mtg.model.MultiVAE.setup_anndata(
            adata,
            categorical_covariate_keys=['Batch'],
            rna_indices_end=4000,
        )
        model = mtg.model.MultiVAE(
            adata, 
            losses=['nb', 'mse'],
            n_layers_encoders=[2, 2],
            n_layers_decoders=[2, 2],
        )
        model.train(max_epochs=args.epochs, lr=args.lr, batch_size=args.bs)
        model.get_latent_representation()
        latent_space = adata.obsm['latent']
        latent_space1 = latent_space[0:sum(batch==1)[0],:]
        latent_space2 = latent_space[sum(batch==1)[0]:,:]
        latent_space_list = [latent_space1, latent_space2]

    elif (file_paths['rna_path'] is None) and (file_paths['adt_path'] is not None) and (file_paths['atac_path'] is not None):
        result, batch = read_h5_data(file_paths["rna_path"],file_paths["adt_path"],file_paths['atac_path'])
        result_list = [] 
        for i in range(len(list(result))):
            temp = []
            temp.append(result[list(result.keys())[i]])
            result_list.append(temp)
        adata = mtg.data.organize_multiome_anndatas(
            adatas = result_list, 
        )
        adata.obs["Batch"] = batch
        mtg.model.MultiVAE.setup_anndata(
            adata,
            categorical_covariate_keys=['Batch'],
            rna_indices_end=4000,
        )
        model = mtg.model.MultiVAE(
            adata, 
            losses=['mse','nb'],
            loss_coefs={'kl': 1e-5,
                       'integ': 0,
                       },
            n_layers_encoders=[2, 2],
            n_layers_decoders=[2, 2],
        )
        model.train(max_epochs=args.epochs, lr=args.lr, batch_size=args.bs)
        model.get_latent_representation()
        latent_space = adata.obsm['latent']
        latent_space1 = latent_space[0:sum(batch==1)[0],:]
        latent_space2 = latent_space[sum(batch==1)[0]:,:]
        latent_space_list = [latent_space1, latent_space2]

    elif (file_paths['rna_path'] is not None) and (file_paths['adt_path'] is None) and (file_paths['atac_path'] is not None):
        result, batch = read_h5_data(file_paths["rna_path"],file_paths["adt_path"],file_paths['atac_path'])
        result_list = [] 
        for i in range(len(list(result))):
            temp = []
            temp.append(result[list(result.keys())[i]])
            result_list.append(temp)
        adata = mtg.data.organize_multiome_anndatas(
            adatas = result_list, 
        )
        adata.obs["Batch"] = batch
        mtg.model.MultiVAE.setup_anndata(
            adata,
            categorical_covariate_keys=['Batch'],
            rna_indices_end=4000,
        )
        model = mtg.model.MultiVAE(
            adata, 
            losses=['nb', 'mse'], #dataset5 nb nb dataset6 nb mse
            loss_coefs={'kl': 1e-5,
                       'integ': 0,
                       },
            n_layers_encoders=[2, 2],
            n_layers_decoders=[2, 2],
        )
        model.train(max_epochs=args.epochs, lr=args.lr, batch_size=args.bs)
        model.get_latent_representation()
        latent_space = adata.obsm['latent']
        latent_space1 = latent_space[0:sum(batch==1)[0],:]
        latent_space2 = latent_space[sum(batch==1)[0]:,:]
        latent_space_list = [latent_space1, latent_space2]

    elif (file_paths['rna_path'] is not None) and (file_paths['adt_path'] is not None) and (file_paths['atac_path'] is not None):
        result, batch = read_h5_data(file_paths["rna_path"],file_paths["adt_path"],file_paths["atac_path"])
        result_list = [] 
        for i in range(len(list(result))):
            temp = []
            temp.append(result[list(result.keys())[i]])
            result_list.append(temp)
        adata = mtg.data.organize_multiome_anndatas(
            adatas = result_list, 
        )
        adata.obs["Batch"] = batch
        mtg.model.MultiVAE.setup_anndata(
            adata,
            categorical_covariate_keys=['Batch'],
            rna_indices_end=4000,
        )
        model = mtg.model.MultiVAE(
            adata, 
            losses=['nb', 'mse', 'mse'],
            loss_coefs={'kl': 1e-5,
                       'integ': 0,
                       },
            #z_dim = 100,
            n_layers_encoders=[2, 2, 2],
            n_layers_decoders=[2, 2, 2],
        )
        model.train(max_epochs=args.epochs, lr=args.lr, batch_size=args.bs)
        model.get_latent_representation()
        latent_space = adata.obsm['latent']
        latent_space1 = latent_space[0:sum(batch==1)[0],:]
        latent_space2 = latent_space[sum(batch==1)[0]:(sum(batch==1)[0]+sum(batch==2)[0]),:]
        latent_space3 = latent_space[(sum(batch==1)[0]+sum(batch==2)[0]):,:]
        latent_space_list = [latent_space1, latent_space2, latent_space3]
    
    return latent_space_list

# run method
if args.path2==[]:
    file_paths = {
        "rna_path": args.path1,
        "adt_path": None,
        "atac_path": args.path3
    }
elif args.path3==[]:
    file_paths = {
        "rna_path": args.path1,
        "adt_path": args.path2,
        "atac_path": None
    }
elif args.path1==[]:
    file_paths = {
        "rna_path": None,
        "adt_path": args.path2,
        "atac_path": args.path3,
    }
else:
    file_paths = {
        "rna_path": args.path1,
        "adt_path": args.path2,
        "atac_path": args.path3
    }
result = run_Multigrate(file_paths)
result = np.concatenate(result,0)
result = np.transpose(result)
end_time = time.time()
all_time = end_time - begin_time
print(all_time)
print(result.shape)

# save results
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result)
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")

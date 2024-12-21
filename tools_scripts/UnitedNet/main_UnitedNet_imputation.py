import os
import sys
import time
import h5py
import torch
import random
import anndata
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from anndata import AnnData
from src.interface import UnitedNet
from src.configs import *

parser = argparse.ArgumentParser("UnitedNet")
parser.add_argument('--data_path', default='NULL', help='path to load the data')
parser.add_argument('--train_fids', metavar='trainid', nargs='+', default=[], help='file ids to train data1')
parser.add_argument('--impute_fids', metavar='imputeid', default='1', help='file ids to train data2')
parser.add_argument('--save_path', default='NULL', help='path to save the output data')
parser.add_argument('--seed',  type=int,  default=1, help='path to save the output data')
args = parser.parse_args()

# This script is designed for UnitedNet cross-integration (imputation).
#python main_UnitedNet_imputation.py --data_path "../../data/dataset_final_imputation_hvg/D56/data1" --train_fids '1' --impute_fids '2' --save_path "../../result/imputation_filter/D56/data1/UnitedNet"
#python main_UnitedNet_imputation.py --data_path "../../data/dataset_final_imputation_hvg/D56/data2" --train_fids '1' --impute_fids '2' --save_path "../../result/imputation_filter/D56/data2/UnitedNet"


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = "cuda:0"

def data_loader(path, bid):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
        feat = [i.decode('utf-8') for i in f['matrix/features']]
        cid = [i.decode('utf-8') for i in f['matrix/barcodes']]
    adata = AnnData(X=X)
    adata.obs['batch'] = str(bid)
    adata.var_names = feat
    adata.obs_names = cid
    return adata

def split_data(test_batch, adata_atac, adata_gex):
    adata_atac_train = adata_atac[adata_atac.obs['batch']!=test_batch]
    adata_atac_test  = adata_atac[adata_atac.obs['batch']==test_batch]
    del adata_atac_test.obs['label']
    adata_gex_train = adata_gex[adata_gex.obs['batch']!=test_batch]
    adata_gex_test  = adata_gex[adata_gex.obs['batch']==test_batch]
    del adata_gex_test.obs['label']
    return [adata_atac_train, adata_gex_train], [adata_atac_test, adata_gex_test]

begin_time = time.time()
device = "cuda:0"

print("----preparing training data..")
rna_list = []
adt_list = []
unique_batches = []
for trainid in args.train_fids:
	unique_batches.append('T'+trainid)
	rna_h5 = os.path.join(args.data_path, 'reference', 'rna'+trainid+'.h5')
	adt_h5 = os.path.join(args.data_path, 'reference', 'atac'+trainid+'.h5')
	lb_csv = os.path.join(args.data_path, 'reference', 'cty'+trainid+'.csv')
	lbs = pd.read_csv(lb_csv)['x'].tolist()
	print("->Loading "+rna_h5)
	genes = data_loader(rna_h5, 'T'+trainid)
	genes.obs['label'] = lbs
	rna_list.append(genes)
	print(genes.X.shape)
	print("->Loading "+adt_h5)
	proteins = data_loader(adt_h5, trainid)
	proteins.obs['label'] = lbs
	adt_list.append(proteins)

print("----preparing testing data..")
unique_batches.append('I'+args.impute_fids)
rna_test_h5 = os.path.join(args.data_path, 'reference', 'rna'+args.impute_fids+'.h5')
adt_test_h5 = os.path.join(args.data_path, 'gt', 'atac'+args.impute_fids+'.h5')
lb_test_csv = os.path.join(args.data_path, 'reference', 'cty'+args.impute_fids+'.csv')
lbs_test = pd.read_csv(lb_test_csv)['x'].tolist()
print("->Loading "+rna_test_h5)
gene_testing = data_loader(rna_test_h5, 'I'+args.impute_fids)
gene_testing.obs['label'] = lbs_test
print("->Loading "+adt_test_h5)
adt_testing = data_loader(adt_test_h5, 'I'+args.impute_fids)
adt_testing.obs['label'] = lbs_test
rna_raw = gene_testing.X.copy()
adt_raw = adt_testing.X.copy()

rna_list.append(gene_testing)
adt_list.append(adt_testing)

print("----combining all data..")
adata_gex= anndata.concat(rna_list) #gene
adata_atac = anndata.concat(adt_list) #atac
test_batch = 'I'+args.impute_fids
print(test_batch)
adatas_train, adatas_test = split_data(test_batch, adata_atac, adata_gex)


atacseq_config = {
    "train_batch_size": 512,
    "finetune_batch_size": 5000,
    "transfer_batch_size": 512,
    "train_epochs": 10,
    "finetune_epochs": 10,
    "transfer_epochs": 20,
    "train_task": "supervised_group_identification",
    "finetune_task": "cross_model_prediction_clas",
    "transfer_task": "supervised_group_identification",
    "train_loss_weight": None,
    "finetune_loss_weight": None,
    "transfer_loss_weight": None,
    "lr": 0.01,
    "checkpoint": 1,
    "n_head": 1,
    "noise_level":[0,0],
    "fuser_type":"WeightedMean",
    "encoders": [
        {
            "input": (adata_atac).shape[1], #13634,
            "hiddens": [64, 64],
            "output": 64,
            "use_biases": [True, True, True],
            "dropouts": [0, 0, 0],
            "activations": ["relu", "relu", "relu"],
            "use_batch_norms": [True, True, True],
            "use_layer_norms": [False, False, False],
            "is_binary_input": False,
        },
        {
            "input": (adata_gex).shape[1], #4000,
            "hiddens": [64, 64],
            "output": 64,
            "use_biases": [True, True, True],
            "dropouts": [0, 0, 0],
            "activations": ["relu", "relu", "relu"],
            "use_batch_norms": [True, True, True],
            "use_layer_norms": [False, False, False],
            "is_binary_input": False,
        },
    ],
    "latent_projector": None,
    "decoders": [
        {
            "input": 64,
            "hiddens": [64, 64],
            "output": (adata_atac).shape[1], #13634,
            "use_biases": [True, True, True],
            "dropouts": [0, 0, 0],
            "activations": ["relu", "relu", None],
            "use_batch_norms": [False, False, False],
            "use_layer_norms": [False, False, False],
        },
        {
            "input": 64,
            "hiddens": [64, 64],
            "output": (adata_gex).shape[1], #4000,
            "use_biases": [True, True, True],
            "dropouts": [0, 0, 0],
            "activations": ["relu", "relu", None],
            "use_batch_norms": [False, False, False],
            "use_layer_norms": [False, False, False],
        },
    ],
    "discriminators": [
        {
            "input": (adata_atac).shape[1], #13634,
            "hiddens": [64],
            "output": 1,
            "use_biases": [True, True],
            "dropouts": [0, 0],
            "activations": ["relu", "sigmoid"],
            "use_batch_norms": [False, False],
            "use_layer_norms": [False, True],
        },
        {
            "input": (adata_gex).shape[1], #4000,
            "hiddens": [64],
            "output": 1,
            "use_biases": [True, True],
            "dropouts": [0, 0],
            "activations": ["relu", "sigmoid"],
            "use_batch_norms": [False, False],
            "use_layer_norms": [False, True],
        },
    ],
    "projectors": {
        "input": 64,
        "hiddens": [],
        "output": 100,
        "use_biases": [True],
        "dropouts": [0],
        "activations": ["relu"],
        "use_batch_norms": [False],
        "use_layer_norms": [True],
    },
    "clusters": {
        "input": 100,
        "hiddens": [],
        "output": len(set(adatas_train[1].obs['label'].tolist())), #23,
        "use_biases": [False],
        "dropouts": [0],
        "activations": [None],
        "use_batch_norms": [False],
        "use_layer_norms": [False],
    },
}

model = UnitedNet(f"{args.save_path}/", device=device, technique=atacseq_config)
model.train(adatas_train, verbose=True)
adatas_prd = model.predict(adatas_test)

end_time = time.time()
all_time = end_time - begin_time
print(all_time)
print(len(adatas_prd))
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
    
print("---Saving data")
file = h5py.File(args.save_path+"/imputed_result_atac.h5",'w')
file.create_dataset("data", data=adatas_prd[0][1])
file = h5py.File(args.save_path+"/within_prediction_atac.h5",'w')
file.create_dataset("data", data=adatas_prd[0][0])
file = h5py.File(args.save_path+"/groundtruth_raw_atac.h5",'w')
file.create_dataset("data", data=adatas_test[0].X)

file1 = h5py.File(args.save_path+"/imputed_result_rna.h5",'w')
file1.create_dataset("data", data=adatas_prd[1][0])
file1 = h5py.File(args.save_path+"/within_prediction_rna.h5",'w')
file1.create_dataset("data", data=adatas_prd[1][1])
file1 = h5py.File(args.save_path+"/groundtruth_raw_rna.h5",'w')
file1.create_dataset("data", data=adatas_test[1].X)


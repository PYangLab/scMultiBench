import os
import sys
import time
import h5py
import random
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from anndata import AnnData
from src.interface import UnitedNet
from src.configs import *


random.seed(1)
parser = argparse.ArgumentParser("UnitedNet")
parser.add_argument('--path1', metavar='DIR', nargs='+', default=[], help='path to train data1')
parser.add_argument('--path2', metavar='DIR', nargs='+', default=[], help='path to train data2')
parser.add_argument('--cty_path', metavar='DIR', nargs='+', default=[], help='path to cty')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

begin_time = time.time()
device = "cuda:0"
def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
    adata = AnnData(X=(X))
    return adata
    
data_path1 = args.path1
data_path2 = args.path2
cty_paths = args.cty_path

adata_rna_list = []
adata_atac_list = []
i = 0
for data_path in data_path1:
    temp = data_loader(data_path)
    temp.obs['batch'] = i
    adata_rna_list.append(temp)
    i = i + 1
    
i = 0
for data_path in data_path2:
    temp = data_loader(data_path)
    temp.obs['batch'] = i
    adata_atac_list.append(temp)
    i = i + 1

adata_gex = sc.concat(adata_rna_list, axis=0, join='inner')
adata_atac = sc.concat(adata_atac_list, axis=0, join='inner')
print(adata_gex)
    
cty_list = []
for cty_path in cty_paths:
    cty_df = pd.read_csv(cty_path, skiprows=1, header=None)
    cell_types = cty_df.iloc[:, 1].values
    cty_list.append(cell_types)
cty = np.concatenate(cty_list)

adata_gex.obs['cell_type'] = cty
adata_atac.obs['cell_type'] = cty
adata_atac.obs['label'] = list(adata_atac.obs['cell_type'])
adata_gex.obs['label'] = list(adata_gex.obs['cell_type'])
adatas_train = [adata_atac, adata_gex]

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
            "input": (adata_atac).shape[1],
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
            "input": (adata_gex).shape[1],
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
            "output": (adata_atac).shape[1], 
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
        "output": len(set(cty)), #23,
        "use_biases": [False],
        "dropouts": [0],
        "activations": [None],
        "use_batch_norms": [False],
        "use_layer_norms": [False],
    },
}


model = UnitedNet("./", device=device, technique=atacseq_config)
model.train(adatas_train, verbose=True)
model.finetune(adatas_train, verbose=True)

adata_fused = model.infer(adatas_train)
result = adata_fused.X

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

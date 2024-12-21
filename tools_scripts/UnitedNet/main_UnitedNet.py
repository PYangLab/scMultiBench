import os
import sys
import time
import h5py
import random
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from src.configs import *
from anndata import AnnData
from src.interface import UnitedNet

parser = argparse.ArgumentParser("UnitedNet")
parser.add_argument('--train_path1', metavar='DIR', nargs='+', default=[], help='path to train RNA')
parser.add_argument('--train_path2', metavar='DIR', nargs='+', default=[], help='path to train ATAC')
parser.add_argument('--test_path1', metavar='DIR', nargs='+', default=[], help='path to test RNA')
parser.add_argument('--test_path2', metavar='DIR', nargs='+', default=[], help='path to test ATAC')
parser.add_argument('--train_cty_path', metavar='DIR', nargs='+', default=[], help='path to cty')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
parser.add_argument('--index', type=int, default=1, help='batch size')
args = parser.parse_args()

# The UnitedNet script is designed for vertical integration of RNA and ATAC data, as well as cross-integration of multiple RNA and ATAC datasets. This is a supervised method and requires labels.
# The output is a joint embedding (dimension reduction) and predicted labels (classification)
# the commond for unitednet (RNA+ATAC)
# python main_UnitedNet.py --train_path1 "./rna1.h5"  --train_path2 "./atac1.h5"   --train_cty_path "./cty1.csv" --test_path1 "./rna.h5"   --test_path2 "./atac.h5" --save_path "./UnitedNet"
# the commond for unitednet (multiple RNA+ATAC)
# python main_UnitedNet.py --train_path1 "./rna1.h5" "./rna2.h5"  --train_path2 "./atac1.h5"  "./atac2.h5"  --train_cty_path "./cty1.csv" "./cty2.csv" --test_path1 "./rna.h5"   --test_path2 "./atac.h5" --save_path "./UnitedNet"

begin_time = time.time()
device = "cuda:0"
def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
    adata = AnnData(X=(X))
    return adata
    
data_path1 = args.train_path1
data_path2 = args.train_path2
cty_paths = args.train_cty_path
test_data_path1 = args.test_path1
test_data_path2 = args.test_path2

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
    
test_adata_rna_list = []
test_adata_atac_list = []
i = 0
for data_path in test_data_path1:
    temp = data_loader(data_path)
    temp.obs['batch'] = i
    test_adata_rna_list.append(temp)
    i = i + 1
    
i = 0
print(test_data_path2,"!!")
for data_path in test_data_path2:
    temp = data_loader(data_path)
    temp.obs['batch'] = i
    test_adata_atac_list.append(temp)
    i = i + 1

adata_gex = sc.concat(adata_rna_list, axis=0, join='inner')
adata_atac = sc.concat(adata_atac_list, axis=0, join='inner')
print(adata_gex)
print(test_adata_rna_list)
print(test_adata_atac_list)

if len(test_adata_rna_list)>1:
    test_adata_gex = sc.concat(test_adata_rna_list, axis=0, join='inner')
    test_adata_atac = sc.concat(test_adata_atac_list, axis=0, join='inner')
    print(test_adata_gex)
else:
    test_adata_gex = test_adata_rna_list[0]
    test_adata_atac = test_adata_atac_list[0]
    
cty_list = []
for cty_path in cty_paths:
    cty_df = pd.read_csv(cty_path, skiprows=1, header=None)
    print(cty_df.iloc)
    cell_types = cty_df.iloc[:, 0].values
    cty_list.append(cell_types)
cty = np.concatenate(cty_list)

adata_gex.obs['cell_type'] = cty
adata_atac.obs['cell_type'] = cty
adata_atac.obs['label'] = list(adata_atac.obs['cell_type'])
adata_gex.obs['label'] = list(adata_gex.obs['cell_type'])
adatas_train = [adata_atac, adata_gex]
adatas_test = [test_adata_atac, test_adata_gex]

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

adata_fused = model.infer(adatas_test)
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
file = h5py.File(args.save_path+"/embedding{}.h5".format(args.index), 'w')
file.create_dataset('data', data=result)
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")

labels_preds_results = model.predict_label(adatas_test)
labels_preds_actual = model.model.label_encoder.inverse_transform(labels_preds_results.tolist())
pre_result_pre=dict()
pre_result_pre['0'] = labels_preds_actual
dt2save_pre = pd.DataFrame(pre_result_pre)
dt2save_pre.to_csv(os.path.join(args.save_path, "predict{}.csv".format(args.index)))

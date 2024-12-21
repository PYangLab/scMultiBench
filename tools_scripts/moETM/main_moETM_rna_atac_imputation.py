import os
import h5py
import torch
import random
import anndata
import warnings
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
from anndata import AnnData
from moETM.build_model import build_moETM
from moETM.train import Trainer_moETM_for_cross_prediction, Train_moETM_for_cross_prediction
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('moETM')
parser.add_argument('--data_path', default='NULL', help='path to load the data')
parser.add_argument('--train_fids', metavar='trainid', nargs='+', default=[], help='file ids to train data1')
parser.add_argument('--impute_fids', metavar='imputeid', default='1', help='file ids to train data2')
parser.add_argument('--save_path', default='NULL', help='path to save the output data')
parser.add_argument('--direction', default='rna_to_another', help='path to save the output data')
parser.add_argument('--seed',  type=int,  default=1, help='path to save the output data')
args = parser.parse_args()

# This script is designed for imputing RNA or ATAC modalities using a multiome reference.
# run example
# python main_moETM_rna_atac_imputation.py --data_path "../../data/dataset_final_imputation_hvg/D56/data1" --train_fids '1' --impute_fids '2' --save_path './' --direction 'rna_to_another'
# python main_moETM_rna_atac_imputation.py --data_path "../../data/dataset_final_imputation_hvg/D56/data1" --train_fids '1' --impute_fids '2' --save_path './' --direction 'another_to_rna'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

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

def prepare_nips_dataset(adata_gex, adata_mod2, unique_batch, batch_col = 'batch'):
    batch_index = np.array(adata_gex.obs[batch_col].values)
    #unique_batch = list(np.unique(batch_index))
    batch_index = np.array([unique_batch.index(xs) for xs in batch_index])
    obs = adata_gex.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)
    adata_gex = ad.AnnData(X=adata_gex.X, obs=obs)
    obs = adata_mod2.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)
    X = adata_mod2.X
    adata_mod2 = ad.AnnData(X=X, obs=obs)
    Index = np.array(X.sum(1)>0).squeeze()
    adata_gex = adata_gex[Index]
    obs = adata_gex.obs
    adata_gex = ad.AnnData(X=adata_gex.X, obs=obs)
    adata_mod2 = adata_mod2[Index]
    obs = adata_mod2.obs
    adata_mod2 = ad.AnnData(X=adata_mod2.X, obs=obs)
    return adata_gex, adata_mod2

def data_process_moETM_cross_prediction(adata_mod1, adata_mod2, test_batch):
    train_adata_mod1 = adata_mod1[adata_mod1.obs['batch']!=test_batch].copy()
    train_adata_mod2 = adata_mod2[adata_mod2.obs['batch']!=test_batch].copy()
    test_adata_mod1 = adata_mod1[adata_mod1.obs['batch']==test_batch].copy()
    test_adata_mod2 = adata_mod2[adata_mod2.obs['batch']==test_batch].copy()

    ########################################################
    # Training dataset
    X_mod1 = np.array(train_adata_mod1.X)
    X_mod2 = np.array(train_adata_mod2.X)
    batch_index = np.array(train_adata_mod1.obs['batch_indices'])

    X_mod1 = X_mod1 / X_mod1.sum(1)[:, np.newaxis]
    X_mod2 = X_mod2 / X_mod2.sum(1)[:, np.newaxis]

    X_mod1_train_T = torch.from_numpy(X_mod1).float()
    X_mod2_train_T = torch.from_numpy(X_mod2).float()
    batch_index_train_T = torch.from_numpy(batch_index).to(torch.int64).cuda()

    # Testing dataset
    X_mod1 = np.array(test_adata_mod1.X)
    X_mod2 = np.array(test_adata_mod2.X)
    batch_index = np.array(test_adata_mod1.obs['batch_indices'])

    sum1 = X_mod1.sum(1)
    sum2 = X_mod2.sum(1)

    X_mod1 = X_mod1 / X_mod1.sum(1)[:, np.newaxis]
    X_mod2 = X_mod2 / X_mod2.sum(1)[:, np.newaxis]

    X_mod1_test_T = torch.from_numpy(X_mod1).float()
    X_mod2_test_T = torch.from_numpy(X_mod2).float()
    batch_index_test_T = torch.from_numpy(batch_index).to(torch.int64)
    del X_mod1, X_mod2, batch_index
    return X_mod1_train_T, X_mod2_train_T, batch_index_train_T, X_mod1_test_T, X_mod2_test_T, batch_index_test_T, test_adata_mod1, train_adata_mod1, sum1, sum2

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
	genes.obs['lbs'] = lbs
	rna_list.append(genes)
	print(genes.X.shape)
	print("->Loading "+adt_h5)
	proteins = data_loader(adt_h5, trainid)
	proteins.obs['lbs'] = lbs
	adt_list.append(proteins)

print("----preparing testing data..")
unique_batches.append('I'+args.impute_fids)
rna_test_h5 = os.path.join(args.data_path, 'reference', 'rna'+args.impute_fids+'.h5')
adt_test_h5 = os.path.join(args.data_path, 'gt', 'atac'+args.impute_fids+'.h5')
lb_test_csv = os.path.join(args.data_path, 'reference', 'cty'+args.impute_fids+'.csv')
lbs_test = pd.read_csv(lb_test_csv)['x'].tolist()
print("->Loading "+rna_test_h5)
gene_testing = data_loader(rna_test_h5, 'I'+args.impute_fids)
gene_testing.obs['lbs'] = lbs_test
print("->Loading "+adt_test_h5)
adt_testing = data_loader(adt_test_h5, 'I'+args.impute_fids)
adt_testing.obs['lbs'] = lbs_test
rna_raw = gene_testing.X.copy()
adt_raw = adt_testing.X.copy()

rna_list.append(gene_testing)
adt_list.append(adt_testing)

print("----combining all data..")
adata_mod1= anndata.concat(rna_list)
adata_mod2 = anndata.concat(adt_list)

print("----preprocessing rna data")
adata_mod1_original = adata_mod1.copy()
sc.pp.normalize_total(adata_mod1, target_sum=1e4)
sc.pp.log1p(adata_mod1)
sc.pp.highly_variable_genes(adata_mod1)
index = adata_mod1.var['highly_variable'].values

#adata_mod1_original = adata_mod1_original[:,index].copy()

print("----preprocessing atac data")
adata_mod2_original = adata_mod2.copy()
sc.pp.normalize_total(adata_mod2, target_sum=1e4)
sc.pp.log1p(adata_mod2)
sc.pp.highly_variable_genes(adata_mod2)
index = adata_mod2.var['highly_variable'].values

#adata_mod2_original = adata_mod2_original[:,index].copy()

adata_mod1, adata_mod2 = prepare_nips_dataset(adata_mod1_original, adata_mod2, unique_batches, 'batch')
n_total_sample = adata_mod1.shape[0]

X_mod1_train_T, X_mod2_train_T, batch_index_train_T, X_mod1_test_T, X_mod2_test_T, batch_index_test_T, test_adata_mod1, train_adata_mod1, test_mod1_sum, test_mod2_sum = data_process_moETM_cross_prediction(adata_mod1, adata_mod2, 'I'+args.impute_fids)

num_batch = len(batch_index_train_T.unique())+1
input_dim_mod1 = X_mod1_train_T.shape[1]
input_dim_mod2 = X_mod2_train_T.shape[1]
train_num = X_mod1_train_T.shape[0]
print("---> Number of training data: "+str(train_num))

num_topic = 200
emd_dim = 400
encoder_mod1, encoder_mod2, decoder, optimizer = build_moETM(input_dim_mod1, input_dim_mod2, num_batch, num_topic=num_topic, emd_dim=emd_dim)

print("****** imputation for "+args.direction)
direction = args.direction
trainer = Trainer_moETM_for_cross_prediction(encoder_mod1, encoder_mod2, decoder, optimizer, direction)
Total_epoch = 500
batch_size = 2000
Train_set = [X_mod1_train_T, X_mod2_train_T, batch_index_train_T]
Test_set = [X_mod1_test_T, X_mod2_test_T, batch_index_test_T, test_adata_mod1, test_mod1_sum, test_mod2_sum]
recon_mod,gt_mod = Train_moETM_for_cross_prediction(trainer, Total_epoch, train_num, batch_size, Train_set, Test_set)

print("---Saving data")
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/imputation.h5",'w')
file.create_dataset("data", data=recon_mod)
file = h5py.File(args.save_path+"/groundtruth_ori.h5",'w')
file.create_dataset("data", data=gt_mod)
if args.direction == 'rna_to_another':
    file = h5py.File(args.save_path+"/groundtruth_norm.h5",'w')
    #file.create_dataset("groundtruth_raw", data= adt_raw)
    file.create_dataset("data", data= np.array(X_mod2_test_T))
else:
    file = h5py.File(args.save_path+"/groundtruth_norm.h5",'w')
    #file.create_dataset("groundtruth_raw", data= rna_raw)
    file.create_dataset("data", data= np.array(X_mod1_test_T))








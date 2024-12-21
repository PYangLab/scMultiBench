import os
import gc
import sys
import time
import h5py
import scipy
import torch
import random
import argparse
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import scipy.io as sio
import scipy.sparse as sp
from eval_utils import evaluate
import torch.nn.functional as F
from torch.autograd import Variable
from utils import calc_weight
from moETM.build_model import build_moETM
from moETM.train import Trainer_moETM, Train_moETM
from dataloader import prepare_nips_dataset

parser = argparse.ArgumentParser("moETM_atac")
parser.add_argument('--path1', metavar='DIR', nargs='+', default=[], help='path to RNA')
parser.add_argument('--path2', metavar='DIR', nargs='+', default=[], help='path to ATAC')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()

# This script is designed for moETM for vertical integration, where the input should be RNA+ATAC.
# run example
# python main_moETM_rna_atac.py --path1 "../../data/dataset_final/D15/rna.h5" --path2 "../../data/dataset_final/D15/atac.h5"  --save_path "../../result/embedding/D15/"

begin_time = time.time()

def load_data_with_batch_and_label(atac_path, rna_path, batch_name):
    with h5py.File(atac_path, 'r') as f:
        data_atac = np.array(f['matrix/data']).T
        barcodes_atac = np.array(f['matrix/barcodes'])
        features_atac = np.array(f['matrix/features'])
    with h5py.File(rna_path, 'r') as f:
        data_rna = np.array(f['matrix/data']).T
        barcodes_rna = np.array(f['matrix/barcodes'])
        features_rna = np.array(f['matrix/features'])
    adata_ATAC = sc.AnnData(X=data_atac, obs=pd.DataFrame(index=barcodes_atac), var=pd.DataFrame(index=features_atac))
    adata_RNA = sc.AnnData(X=data_rna, obs=pd.DataFrame(index=barcodes_rna), var=pd.DataFrame(index=features_rna))
    adata_ATAC.obs['batch'] = batch_name
    adata_RNA.obs['batch'] = batch_name
    return adata_ATAC, adata_RNA

def get_adata_mod(atac_files, rna_files, batch_names):
    atac_data_list = []
    rna_data_list = []
    for atac, rna,  batch in zip(atac_files, rna_files,  batch_names):
        atac_data, rna_data = load_data_with_batch_and_label(atac, rna,  batch)
        atac_data_list.append(atac_data)
        rna_data_list.append(rna_data)
    adata_ATAC_combined = sc.concat(atac_data_list, axis=0, join='outer')
    adata_RNA_combined = sc.concat(rna_data_list, axis=0, join='outer')



    adata_ATAC_combined_original = ad.AnnData.copy(adata_ATAC_combined)
    adata_RNA_combined_original = ad.AnnData.copy(adata_RNA_combined)

    sc.pp.normalize_total(adata_ATAC_combined, target_sum=1e4)
    sc.pp.log1p(adata_ATAC_combined)
    sc.pp.highly_variable_genes(adata_ATAC_combined)  

    index = adata_ATAC_combined.var['highly_variable'].values
    adata_ATAC_combined = ad.AnnData.copy(adata_ATAC_combined_original)
    adata_ATAC_combined = adata_ATAC_combined[:, index].copy()

    del adata_ATAC_combined_original
    gc.collect()


    sc.pp.normalize_total(adata_RNA_combined, target_sum=1e4)
    sc.pp.log1p(adata_RNA_combined)
    sc.pp.highly_variable_genes(adata_RNA_combined)


    index = adata_RNA_combined.var['highly_variable'].values
    adata_RNA_combined = ad.AnnData.copy(adata_RNA_combined_original)
    adata_RNA_combined = adata_RNA_combined[:, index].copy()

    del adata_RNA_combined_original
    gc.collect()

    adata_mod1, adata_mod2 = prepare_nips_dataset(adata_RNA_combined, adata_ATAC_combined)
    n_total_sample = adata_mod1.shape[0]
    return adata_mod1, adata_mod2,n_total_sample

def Train_moETM(trainer, Total_epoch, train_num, batch_size, Train_set, Test_set, Eval_kwargs):
    LIST = list(np.arange(0, train_num))
    X_mod1, X_mod2, batch_index = Train_set
    test_X_mod1, test_X_mod2, batch_index_test, test_adate = Test_set

    EPOCH = []
    ARI = []
    NMI = []
    ASW = []
    ASW_2 = []
    B_kBET = []
    B_ASW = []
    B_GS = []
    B_ebm = []

    best_ari = 0

    for epoch in range(Total_epoch):
        Loss_all = 0
        NLL_all_mod1 = 0
        NLL_all_mod2 = 0
        KL_all = 0

        tstart = time.time()

        np.random.shuffle(LIST)
        KL_weight = calc_weight(epoch, Total_epoch, 0, 1 / 3, 0, 1e-4)

        for iteration in range(train_num // batch_size):
            x_minibatch_mod1_T = X_mod1[LIST[iteration * batch_size: (iteration + 1) * batch_size], :].to('cuda')
            x_minibatch_mod2_T = X_mod2[LIST[iteration * batch_size: (iteration + 1) * batch_size], :].to('cuda')
            batch_minibatch_T = batch_index[LIST[iteration * batch_size: (iteration + 1) * batch_size]].to('cuda')

            loss, nll_mod1, nll_mod2, kl = trainer.train(x_minibatch_mod1_T, x_minibatch_mod2_T, batch_minibatch_T, KL_weight)

            Loss_all += loss
            NLL_all_mod1 += nll_mod1
            NLL_all_mod2 += nll_mod2
            KL_all += kl

        if (epoch % 10 == 0):

            trainer.encoder_mod1.to('cpu')
            trainer.encoder_mod2.to('cpu')

            embed = trainer.get_embed(test_X_mod1, test_X_mod2)
            test_adate.obsm.update(embed)
            #Result = evaluate(adata=test_adate, n_epoch=epoch, return_fig=True, **Eval_kwargs)
            tend = time.time()

            trainer.encoder_mod1.cuda()
            trainer.encoder_mod2.cuda()
    return embed


def integration_moETM(atac_files, rna_files, batch_names,Total_epoch=500, batch_size=2000):
    adata_mod1, adata_mod2,n_total_sample = get_adata_mod(atac_files, rna_files, batch_names)
    Eval_kwargs = {}
    Eval_kwargs['plot_dir'] = 'result_fig'

        # Ensure the directory exists
    if not os.path.exists(Eval_kwargs['plot_dir']):
        os.makedirs(Eval_kwargs['plot_dir'])
    # Evaluation parameters
    Eval_kwargs['batch_col'] = 'batch_indices'
    Eval_kwargs['plot_fname'] = 'moETM_delta'
    Eval_kwargs['cell_type_col'] = 'cell_type'
    Eval_kwargs['clustering_method'] = 'louvain'
    Eval_kwargs['resolutions'] = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
    Eval_kwargs['plot_dir'] = 'result_fig'
    train_adata_mod1 = adata_mod1
    train_adata_mod2 = adata_mod2

    ########################################################
    # Training dataset
    if isinstance(train_adata_mod1.X, np.ndarray):
        X_mod1 = train_adata_mod1.X
    else:
        X_mod1 = np.array(train_adata_mod1.X.todense())

    if isinstance(train_adata_mod2.X, np.ndarray):
        X_mod2 = train_adata_mod2.X
    else:
        X_mod2 = np.array(train_adata_mod2.X.todense())

    batch_index = np.array(train_adata_mod1.obs['batch_indices'])

    X_mod1 = X_mod1 / X_mod1.sum(1)[:, np.newaxis]
    X_mod2 = X_mod2 / X_mod2.sum(1)[:, np.newaxis]

    X_mod1_train_T = torch.from_numpy(X_mod1).float()
    X_mod2_train_T = torch.from_numpy(X_mod2).float()
    batch_index_train_T = torch.from_numpy(batch_index).to(torch.int64)
    num_batch = len(batch_index_train_T.unique())
    input_dim_mod1 = X_mod1_train_T.shape[1]
    input_dim_mod2 = X_mod2_train_T.shape[1]
    train_num = X_mod1_train_T.shape[0]
    num_topic = 100
    emd_dim = 400
    encoder_mod1, encoder_mod2, decoder, optimizer = build_moETM(input_dim_mod1, input_dim_mod2, num_batch, num_topic=num_topic, emd_dim=emd_dim)
    trainer = Trainer_moETM(encoder_mod1, encoder_mod2, decoder, optimizer)
    
    Train_set = [X_mod1_train_T, X_mod2_train_T, batch_index_train_T]
    Test_set = [X_mod1_train_T, X_mod2_train_T, batch_index_train_T, train_adata_mod1]

    result = Train_moETM(trainer, Total_epoch, train_num, batch_size, Train_set, Test_set, Eval_kwargs)
    return result

# run methods
atac_files = args.path2
rna_files = args.path1
batch_names = ["batch{}".format(i) for i in range(1)]
result = integration_moETM(atac_files, rna_files,batch_names)
end_time = time.time()
all_time = end_time - begin_time
print(all_time)
print(result['delta'].shape)

# save results
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=result['delta'])
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")


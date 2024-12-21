import os
import time
import torch
import random
import argparse

import numpy as np
import pandas as pd
from captum.attr import *

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable

from learn.model import CiteAutoencoder_CITEseq, CiteAutoencoder_SHAREseq, CiteAutoencoder_TEAseq
from learn.train import train_model
from util import setup_seed, MyDataset,ToTensor, real_label, read_h5_data, read_fs_label, get_vae_simulated_data_from_sampling, get_encodings, compute_zscore, compute_log2,save_checkpoint

parser = argparse.ArgumentParser("Matilda")
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--augmentation', type=bool, default= True, help='if augmentation or not')

############# for data build ##############
parser.add_argument('--rna', metavar='DIR',  nargs='+', default=[],  help='path to train rna data')
parser.add_argument('--adt', metavar='DIR',  nargs='+', default=[],  help='path to train adt data')
parser.add_argument('--atac', metavar='DIR', nargs='+', default=[],  help='path to train atac data')
parser.add_argument('--cty', metavar='DIR',  nargs='+', default=[],  help='path to train cell type label')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save results')
parser.add_argument('--device', type=str, default= "cuda", help='cpu or gpu')

##############  for training #################
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
parser.add_argument('--lr', type=float, default=0.02, help='init learning rate')

############# for model build ##############
parser.add_argument('--z_dim', type=int, default=100, help='the number of neurons in latent space')
parser.add_argument('--hidden_rna', type=int, default=185, help='the number of neurons for RNA layer')
parser.add_argument('--hidden_adt', type=int, default=30, help='the number of neurons for ADT layer')
parser.add_argument('--hidden_atac', type=int, default=185, help='the number of neurons for ATAC layer')

args = parser.parse_args()
begin_time = time.time()

# this script is designed for vertical integration (RNA+ADT, RNA+ATAC, RNA+ADT+ATAC).
# we need to train the model first
# python main_matilda_train.py --rna "../../data/dataset_final_cross_validation/D3/rna2.h5" --adt "../../data/dataset_final_cross_validation/D3/adt2.h5"  --cty "../../data/dataset_final_cross_validation/D3/cty2.csv"  --save_path "../../result/embedding/vertical integration/D3/Matilda"
# for dimension reduction, we need to set the dim_reduce parameter as True
# python main_matilda_task.py --rna  "../../data/dataset_final_cross_validation/D3/rna1.h5" --adt  "../../data/dataset_final_cross_validation/D3/adt1.h5" --cty  "../../data/dataset_final_cross_validation/D3/cty1.csv"  --save_path "../../result/embedding/vertical integration/D3/Matilda" --dim_reduce True
# for feature selection, we need to set the fs parameter as True
# python main_matilda_task.py --rna  "../../data/dataset_final_cross_validation/D3/rna1.h5" --adt  "../../data/dataset_final_cross_validation/D3/adt1.h5" --cty  "../../data/dataset_final_cross_validation/D3/cty1.csv"  --save_path "../../result/embedding/vertical integration/D3/Matilda" --fs True

setup_seed(args.seed) ### set random seed in order to reproduce the result
cuda = True if args.device=="cuda" else False
device = torch.device("cuda" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

print()
if args.adt and args.atac:
    mode = "TEAseq"
    train_rna_data_path = args.rna
    train_adt_data_path = args.adt
    train_atac_data_path = args.atac
    train_label_path = args.cty
    
    train_rna_data_list = []
    for rna_path in train_rna_data_path:
        temp = read_h5_data(rna_path, args.device)
        train_rna_data_list.append(temp)
    train_rna_data = torch.cat(train_rna_data_list, dim=0)
    
    train_adt_data_list = []
    for adt_path in train_adt_data_path:
        temp = read_h5_data(adt_path, args.device)
        train_adt_data_list.append(temp)
    train_adt_data = torch.cat(train_adt_data_list, dim=0)
    
    train_atac_data_list = []
    for atac_path in train_atac_data_path:
        temp = read_h5_data(atac_path, args.device)
        train_atac_data_list.append(temp)
    train_atac_data = torch.cat(train_atac_data_list, dim=0)

    cty_list = []
    for cty_path in train_label_path:
        cty_df = pd.read_csv(cty_path, skiprows=1, header=None)
        cell_types = pd.Categorical(cty_df.iloc[:, 0]).codes
        cty_list.append(cell_types)
    train_label = np.concatenate(cty_list)
    train_label = torch.from_numpy(train_label)#
    train_label = train_label.type(LongTensor)

    classify_dim = (max(train_label)+1).cpu().numpy()
    nfeatures_rna = train_rna_data.shape[1]
    nfeatures_adt = train_adt_data.shape[1]
    nfeatures_atac = train_atac_data.shape[1]
    feature_num = nfeatures_rna + nfeatures_adt + nfeatures_atac
    train_rna_data = compute_log2(train_rna_data)
    train_adt_data = compute_log2(train_adt_data)
    train_atac_data = compute_log2(train_atac_data)
    train_rna_data = compute_zscore(train_rna_data)
    train_adt_data = compute_zscore(train_adt_data)
    train_atac_data = compute_zscore(train_atac_data)
    train_data = torch.cat((train_rna_data,train_adt_data,train_atac_data),1)       
    train_transformed_dataset = MyDataset(train_data, train_label)
    train_dl = DataLoader(train_transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

if not args.adt and args.atac:
    mode = "SHAREseq"
    train_rna_data_path = args.rna
    train_atac_data_path = args.atac
    train_label_path = args.cty
    train_rna_data_list = []
    for rna_path in train_rna_data_path:
        temp = read_h5_data(rna_path, args.device)
        train_rna_data_list.append(temp)
    train_rna_data = torch.cat(train_rna_data_list, dim=0)
    
    train_atac_data_list = []
    for adt_path in train_atac_data_path:
        temp = read_h5_data(adt_path, args.device)
        train_atac_data_list.append(temp)
    train_atac_data = torch.cat(train_atac_data_list, dim=0)

    cty_list = []
    for cty_path in train_label_path:
        cty_df = pd.read_csv(cty_path, skiprows=1, header=None)
        cell_types = pd.Categorical(cty_df.iloc[:, 0]).codes
        cty_list.append(cell_types)
    train_label = np.concatenate(cty_list)
    train_label = torch.from_numpy(train_label)#
    train_label = train_label.type(LongTensor)
    classify_dim = (max(train_label)+1).cpu().numpy()
    nfeatures_rna = train_rna_data.shape[1]
    nfeatures_atac = train_atac_data.shape[1]
    feature_num = nfeatures_rna + nfeatures_atac
    train_rna_data = compute_log2(train_rna_data)
    train_atac_data = compute_log2(train_atac_data)
    train_rna_data = compute_zscore(train_rna_data)
    train_atac_data = compute_zscore(train_atac_data)
    train_data = torch.cat((train_rna_data,train_atac_data),1)
    train_transformed_dataset = MyDataset(train_data, train_label)
    train_dl = DataLoader(train_transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    
if args.adt and not args.atac:
    mode = "CITEseq"
    train_rna_data_path = args.rna
    train_adt_data_path = args.adt
    train_label_path = args.cty
    train_rna_data_list = []

    train_adt_data_list = []
    for adt_path in train_adt_data_path:
        temp = read_h5_data(adt_path, args.device)
        train_adt_data_list.append(temp)
    train_adt_data = torch.cat(train_adt_data_list, dim=0)
    
    train_rna_data_list = []
    for rna_path in train_rna_data_path:
        temp = read_h5_data(rna_path, args.device)
        train_rna_data_list.append(temp)
    train_rna_data = torch.cat(train_rna_data_list, dim=0)

    cty_list = []
    for cty_path in train_label_path:
        cty_df = pd.read_csv(cty_path, skiprows=1, header=None)
        cell_types = pd.Categorical(cty_df.iloc[:, 0]).codes
        cty_list.append(cell_types)
        
    train_label = np.concatenate(cty_list)
    train_label = torch.from_numpy(train_label)#
    train_label = train_label.type(LongTensor)
    
    classify_dim = (max(train_label)+1).cpu().numpy()
    nfeatures_rna = train_rna_data.shape[1]
    nfeatures_adt = train_adt_data.shape[1]
    feature_num = nfeatures_rna + nfeatures_adt
    train_rna_data = compute_log2(train_rna_data)
    train_adt_data = compute_log2(train_adt_data)
    train_rna_data = compute_zscore(train_rna_data)
    train_adt_data = compute_zscore(train_adt_data)
    train_data = torch.cat((train_rna_data,train_adt_data),1)
    train_transformed_dataset = MyDataset(train_data, train_label)
    train_dl = DataLoader(train_transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
		
test_dl = "NULL"

print("The dataset is", mode)    
output_v = []
model_save_path = "../trained_model/{}/".format(mode)   
model_save_path_1stage = "../trained_model/{}/simulation_".format(mode)    
save_fs_eachcell = "../output/marker/{}/".format(mode)   

#######build model#########
if mode == "CITEseq":
	model = CiteAutoencoder_CITEseq(nfeatures_rna, nfeatures_adt, args.hidden_rna, args.hidden_adt, args.z_dim, classify_dim)
elif mode == "SHAREseq":
	model = CiteAutoencoder_SHAREseq(nfeatures_rna, nfeatures_atac, args.hidden_rna, args.hidden_atac, args.z_dim, classify_dim)
elif mode == "TEAseq":
	model = CiteAutoencoder_TEAseq(nfeatures_rna, nfeatures_adt, nfeatures_atac, args.hidden_rna, args.hidden_adt, args.hidden_atac, args.z_dim, classify_dim)

#model = nn.DataParallel(model).to(device) #multi gpu
print(args.device,"!#########")
model = model.to(device) #one gpu
########train model#########
model, acc1, num1, train_num = train_model(model, train_dl, test_dl, lr=args.lr, epochs=args.epochs, classify_dim = classify_dim, best_top1_acc=0, save_path=model_save_path,feature_num=feature_num, device=device)
##################prepare to do augmentation##################
if args.augmentation == True:
    stage1_list = []
    for i in np.arange(0, classify_dim):
        stage1_list.append([i, train_num[i]])
        stage1_df = pd.DataFrame(stage1_list)
    if classify_dim%2==0:
        train_median = np.sort(train_num)[int(classify_dim/2)-1]
    else: 
        train_median = np.median(train_num)
    median_anchor = stage1_df[stage1_df[1] == train_median][0]
    train_major = stage1_df[stage1_df[1] > train_median]
    train_minor = stage1_df[stage1_df[1] < train_median]
    anchor_fold = np.array((train_median)/(train_minor[:][1]))
    minor_anchor_cts = train_minor[0].to_numpy()
    major_anchor_cts = train_major[0].to_numpy()

    print(np.array(median_anchor)[0],"!!!!!")
    index = (train_label == int(np.array(median_anchor)[0])).nonzero(as_tuple=True)[0]
    anchor_data = train_data[index.tolist(),:]
    anchor_label = train_label[index.tolist()]
    new_data = anchor_data 
    new_label = anchor_label

    ##############random downsample major cell types##############
    j=0
    for anchor in major_anchor_cts:     
        anchor_num = np.array(train_major[1])[j]
        N = range(anchor_num)
        ds_index = random.sample(N,int(train_median))
        index = (train_label == anchor).nonzero(as_tuple=True)[0]
        anchor_data = train_data[index.tolist(),:]
        anchor_label = train_label[index.tolist()]
        anchor_data = anchor_data[ds_index,:]
        anchor_label = anchor_label[ds_index]
        new_data = torch.cat((new_data,anchor_data),0)
        new_label = torch.cat((new_label,anchor_label.to(args.device)),0)
        j = j+1

    ###############augment for minor cell types##################
    j = 0
    for anchor in minor_anchor_cts:
        aug_fold = int((anchor_fold[j]))    
        remaining_cell = int(train_median - (int(anchor_fold[j]))*np.array(train_minor[1])[j])
        index = (train_label == anchor).nonzero(as_tuple=True)[0]
        anchor_data = train_data[index.tolist(),:]
        anchor_label = train_label[index.tolist()]
        anchor_transfomr_dataset = MyDataset(anchor_data, anchor_label)
        anchor_dl = DataLoader(anchor_transfomr_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)
        reconstructed_data, reconstructed_label, real_data = get_vae_simulated_data_from_sampling(model, anchor_dl, args.device)
        reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
        reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
        reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)

        new_data = torch.cat((new_data,reconstructed_data),0)
        new_label = torch.cat((new_label.to(device), reconstructed_label.to(device)),0)
        for i in range(aug_fold-1):
            reconstructed_data, reconstructed_label,real_data = get_vae_simulated_data_from_sampling(model, anchor_dl, args.device)
            reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
            reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
            reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)
            new_data = torch.cat((new_data,reconstructed_data),0)
            new_label = torch.cat((new_label.to(device),reconstructed_label.to(device)),0)

        reconstructed_data, reconstructed_label,real_data = get_vae_simulated_data_from_sampling(model, anchor_dl, args.device)
        reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
        reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
        reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)

        #add remaining cell
        N = range(np.array(train_minor[1])[j])
        ds_index = random.sample(N, remaining_cell)
        reconstructed_data = reconstructed_data[ds_index,:]
        reconstructed_label = reconstructed_label[ds_index]
        new_data = torch.cat((new_data,reconstructed_data),0)
        new_label = torch.cat((new_label.to(device),reconstructed_label.to(device)),0)
        j = j+1               


#######load the model#########
#######build model#########
if mode == "CITEseq":
	model = CiteAutoencoder_CITEseq(nfeatures_rna, nfeatures_adt, args.hidden_rna, args.hidden_adt, args.z_dim, classify_dim)
elif mode == "SHAREseq":
	model = CiteAutoencoder_SHAREseq(nfeatures_rna, nfeatures_atac, args.hidden_rna, args.hidden_atac, args.z_dim, classify_dim)
elif mode == "TEAseq":
	model = CiteAutoencoder_TEAseq(nfeatures_rna, nfeatures_adt, nfeatures_atac, args.hidden_rna, args.hidden_adt, args.hidden_atac, args.z_dim, classify_dim)

#model = nn.DataParallel(model).to(device) #multi gpu
model = model.to(device) #one gpu

############process new data after augmentation###########
train_transformed_dataset = MyDataset(new_data, new_label)
train_dl = DataLoader(train_transformed_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)

############## train model ###########
model,acc2,num1,train_num = train_model(model, train_dl, test_dl, lr=args.lr, epochs=int(args.epochs/2),classify_dim=classify_dim,best_top1_acc=0, save_path=model_save_path,feature_num=feature_num, device=device)
checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
if os.path.exists(checkpoint_tar):
    checkpoint = torch.load(checkpoint_tar)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print("load successfully")
model,acc2,num1,train_num = train_model(model, train_dl, test_dl, lr=args.lr/10, epochs=int(args.epochs/2),classify_dim=classify_dim,best_top1_acc=0, save_path=model_save_path,feature_num=feature_num, device=device)


end_time = time.time()
all_time = end_time - begin_time
print(all_time)
 
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")
    

pd.DataFrame([classify_dim]).to_csv("classify_dim.csv", index=False, header=False)


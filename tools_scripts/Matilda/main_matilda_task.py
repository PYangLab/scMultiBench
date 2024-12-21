import os
import random
import argparse
import numpy as np
import pandas as pd
from captum.attr import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable

from learn.model import CiteAutoencoder_CITEseq, CiteAutoencoder_SHAREseq, CiteAutoencoder_TEAseq
from learn.train import train_model
from learn.predict import test_model
from util import setup_seed, MyDataset,ToTensor, read_h5_data, read_fs_label, get_vae_simulated_data_from_sampling, get_encodings, compute_zscore, compute_log2
import h5py,scipy

parser = argparse.ArgumentParser("Matilda")
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--classification', type=bool, default= False, help='if augmentation or not')
parser.add_argument('--query', type=bool, default= False, help='if the data is query of reference')
parser.add_argument('--fs', type=bool, default= False, help='if doing feature selection or not')
parser.add_argument('--fs_method', type=str, default= "IntegratedGradient", help='choose the feature selection method')
parser.add_argument('--dim_reduce', type=bool, default= False, help='save latent space')
parser.add_argument('--simulation', type=bool, default= False, help='save simulation result')
parser.add_argument('--simulation_ct', type=str, default= "CD16 Mono", help='save simulation result') # if simulation_ct is -1, genarate all cells
parser.add_argument('--simulation_num', type=int, default= 100, help='save simulation result')

############# for data build ##############
parser.add_argument('--rna', metavar='DIR', default='NULL', help='path to train rna data')
parser.add_argument('--adt', metavar='DIR', default='NULL', help='path to train adt data')
parser.add_argument('--atac', metavar='DIR', default='NULL', help='path to train atac data')
parser.add_argument('--cty', metavar='DIR', default='NULL', help='path to train cell type label')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save results')
parser.add_argument('--device', type=str, default= "cuda", help='cpu or gpu')
##############  for training #################
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--classify_dim', type=int, default=17, help='batch size')
parser.add_argument('--index', type=int, default=1, help='batch size')

############# for model build ##############
parser.add_argument('--z_dim', type=int, default=100, help='the number of neurons in latent space')
parser.add_argument('--hidden_rna', type=int, default=185, help='the number of neurons for RNA layer')
parser.add_argument('--hidden_adt', type=int, default=30, help='the number of neurons for ADT layer')
parser.add_argument('--hidden_atac', type=int, default=185, help='the number of neurons for ATAC layer')

args = parser.parse_args()
setup_seed(args.seed) ### set random seed in order to reproduce the result
cuda = True if args.device=="cuda" else False
device = torch.device("cuda" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

if args.query:
    path = "query"
else:
    path = "reference"
    
if args.adt != "NULL" and args.atac != "NULL":
    mode = "TEAseq"
    rna_data_path = args.rna
    adt_data_path = args.adt
    atac_data_path = args.atac
    label_path = args.cty
    rna_data = read_h5_data(rna_data_path, args.device)
    adt_data = read_h5_data(adt_data_path, args.device)
    atac_data = read_h5_data(atac_data_path, args.device)
    if label_path == "NULL":
        label = torch.zeros(rna_data.shape[0]).to(args.device)
    else:
        label = read_fs_label(label_path, args.device)
    nfeatures_rna = rna_data.shape[1]
    nfeatures_adt = adt_data.shape[1]
    nfeatures_atac = atac_data.shape[1]
    feature_num = nfeatures_rna + nfeatures_adt + nfeatures_atac
    rna_data = compute_log2(rna_data)
    adt_data = compute_log2(adt_data)
    atac_data = compute_log2(atac_data)
    rna_data = compute_zscore(rna_data)
    adt_data = compute_zscore(adt_data)
    atac_data = compute_zscore(atac_data)
    data = torch.cat((rna_data,adt_data,atac_data),1)       
    transformed_dataset = MyDataset(data, label)
    dl = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

if args.adt == "NULL" and args.atac != "NULL":
    mode = "SHAREseq"
    rna_data_path = args.rna
    atac_data_path = args.atac
    label_path = args.cty
    rna_data = read_h5_data(rna_data_path, args.device)
    atac_data = read_h5_data(atac_data_path, args.device)
    if label_path == "NULL":
        label = torch.zeros(rna_data.shape[0]).to(args.device)
    else:
        label = read_fs_label(label_path, args.device)
    nfeatures_rna = rna_data.shape[1]
    nfeatures_atac = atac_data.shape[1]
    feature_num = nfeatures_rna  + nfeatures_atac
    rna_data = compute_log2(rna_data)
    atac_data = compute_log2(atac_data)
    rna_data = compute_zscore(rna_data)
    atac_data = compute_zscore(atac_data)
    data = torch.cat((rna_data,atac_data),1)
    transformed_dataset = MyDataset(data, label)
    dl = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    
if args.adt != "NULL" and args.atac == "NULL":
    mode = "CITEseq"
    rna_data_path = args.rna
    adt_data_path = args.adt
    label_path = args.cty
    rna_data = read_h5_data(rna_data_path, args.device)
    adt_data = read_h5_data(adt_data_path, args.device)
    if label_path == "NULL":
        label = torch.zeros(rna_data.shape[0]).to(args.device)
    else:
        label = read_fs_label(label_path, args.device)
    nfeatures_rna = rna_data.shape[1]
    nfeatures_adt = adt_data.shape[1]
    feature_num = nfeatures_rna + nfeatures_adt
    rna_data = compute_log2(rna_data)
    adt_data = compute_log2(adt_data)
    rna_data = compute_zscore(rna_data)
    adt_data = compute_zscore(adt_data)
    data = torch.cat((rna_data,adt_data),1)
    transformed_dataset = MyDataset(data, label)
    dl = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    
print("The dataset is", mode)    
output_v = []
model_save_path = "../trained_model/{}/".format(mode)    
save_fs_eachcell = "../output/marker/{}/{}/".format(mode,path)   


output_v = []

rna_name  = h5py.File(rna_data_path,"r")['matrix/features'][:]
if args.adt != "NULL":
    adt_name  = h5py.File(adt_data_path,"r")['matrix/features'][:]
if args.atac!= "NULL":
    atac_name  = h5py.File(atac_data_path,"r")['matrix/features'][:]

#transform_real_label = real_label(args.classify_dim, classify_dim)
#real_cty_df = pd.read_csv("real_cty.csv", header=None)
#transform_real_label = real_cty_df[0].tolist()
classify_dim = pd.read_csv("classify_dim.csv", header=None).values[0, 0]
#classify_dim=23
print(classify_dim, "classify_dim!!!!!!!")
#######build model#########
if mode == "CITEseq":
    model = CiteAutoencoder_CITEseq(nfeatures_rna, nfeatures_adt, args.hidden_rna, args.hidden_adt, args.z_dim, classify_dim)
elif mode == "SHAREseq":
    model = CiteAutoencoder_SHAREseq(nfeatures_rna, nfeatures_atac, args.hidden_rna, args.hidden_atac, args.z_dim, classify_dim)
elif mode == "TEAseq":
    model = CiteAutoencoder_TEAseq(nfeatures_rna, nfeatures_adt, nfeatures_atac, args.hidden_rna, args.hidden_adt, args.hidden_atac, args.z_dim, classify_dim)

#model = nn.DataParallel(model).to(device) #multi gpu
model = model.to(args.device) #one gpu
model.eval()
########train model#########

if args.classification == True:  
    if not os.path.exists('../output/classification/{}/{}'.format(mode,path)):
        os.makedirs('../output/classification/{}/{}'.format(mode,path))
    save_path = open('../output/classification/{}/{}/accuracy_each_ct.txt'.format(mode,path),"w")
    save_path1 = open('../output/classification/{}/{}/accuracy_each_cell.txt'.format(mode,path),"w")
    checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    model, acc1, num1,classified_label, groundtruth_label,prob = test_model(model, dl, transform_real_label, classify_dim = classify_dim, save_path = save_path)
    for j in range(data.shape[0]):
        if args.cty!="NULL":
            print('cell ID: ',j, '\t', '\t', 'real cell type:', groundtruth_label[j], '\t', '\t', 'predicted cell type:', classified_label[j], '\t', '\t', 'probability:', round(float(prob[j]),2), file = save_path1)
        else:
            print('cell ID: ',j, '\t', '\t',  'predicted cell type:', classified_label[j], '\t', '\t', 'probability:', round(float(prob[j]),2), file = save_path1)


if (args.simulation == True) and (args.simulation_ct!="-1"):
    print("simulate celltype:", args.simulation_ct) #transform_real_label[args.simulation_ct]
    checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    if not os.path.exists('../output/simulation_result/{}/{}/'.format(mode,path)):
        os.makedirs('../output/simulation_result/{}/{}/'.format(mode,path))        
        
    sim_cty_index = transform_real_label.index(args.simulation_ct)
    index = (label == sim_cty_index).nonzero(as_tuple=True)[0]
    aug_fold = int(args.simulation_num/int(index.size(0)))    
    remaining_cell = int(args.simulation_num - aug_fold*int(index.size(0)))

    index = (label == sim_cty_index).nonzero(as_tuple=True)[0]
    anchor_data = data[index.tolist(),:]
    anchor_label = label[index.tolist()]
    anchor_transform_dataset = MyDataset(anchor_data, anchor_label)
    anchor_dl = DataLoader(anchor_transform_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)
    new_data = []
    new_label = []
    
    if aug_fold >= 1:
        reconstructed_data, reconstructed_label, real_data = get_vae_simulated_data_from_sampling(model, anchor_dl)
        reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
        reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
        reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)

        new_data = reconstructed_data 
        new_label = reconstructed_label
        for i in range(aug_fold-1):
            reconstructed_data, reconstructed_label,real_data = get_vae_simulated_data_from_sampling(model, anchor_dl)
            reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
            reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
            reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)
            new_data = torch.cat((new_data,reconstructed_data),0)
            new_label = torch.cat((new_label,reconstructed_label.to(device)),0)

    reconstructed_data, reconstructed_label,real_data = get_vae_simulated_data_from_sampling(model, anchor_dl)
    reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
    reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
    reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)

    #add remaining cell
    N = range(np.array(reconstructed_data.size(0)))
    ds_index = random.sample(N, remaining_cell)
    reconstructed_data = reconstructed_data[ds_index,:]
    reconstructed_label = reconstructed_label[ds_index]
    if aug_fold ==0:
        new_data = reconstructed_data
        new_label = reconstructed_label
    else:
        new_data = torch.cat((new_data,reconstructed_data),0)
        new_label = torch.cat((new_label,reconstructed_label.to(device)),0)

    index = (label != sim_cty_index).nonzero(as_tuple=True)[0]
    anchor_data = data[index.tolist(),:]
    anchor_label = label[index.tolist()]
    real_data = data
    real_label = label
    sim_data = torch.cat((anchor_data,new_data),0)
    sim_label = torch.cat((anchor_label,new_label.to(device)),0)
    sim_data_rna = sim_data[:, 0:nfeatures_rna]
    real_data_rna = real_data[:, 0:nfeatures_rna]   
    if mode == "CITEseq":
        sim_data_adt = sim_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
        real_data_adt = real_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
    elif mode == "SHAREseq":
        sim_data_atac = sim_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_atac)]
        real_data_atac = real_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_atac)]
    elif mode == "TEAseq":
        sim_data_adt = sim_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
        sim_data_atac = sim_data[:, (nfeatures_rna+nfeatures_adt):(nfeatures_rna+nfeatures_adt+nfeatures_atac)]
        real_data_adt = real_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
        real_data_atac = real_data[:, (nfeatures_rna+nfeatures_adt):(nfeatures_rna+nfeatures_adt+nfeatures_atac)]

    rna_name_new = []
    adt_name_new = []
    atac_name_new = []
    
    b_list = range(0, real_data_rna.size(0))
    cell_name_real = ['cell_{}'.format(b) for b in b_list]
    b_list = range(0, sim_data_rna.size(0))
    cell_name_sim = ['cell_{}'.format(b) for b in b_list]
    sim_label_new = []
    real_label_new = []
    for j in range(sim_data_rna.size(0)):
        sim_label_new.append(transform_real_label[sim_label[j]])
    for j in range(real_data_rna.size(0)):    
        real_label_new.append(transform_real_label[real_label[j]])
    for i in range(sim_data_rna.size(1)):
        a = str(rna_name[i],encoding="utf-8")
        rna_name_new.append(a)           
        
    if mode == "CITEseq":
        for i in range(sim_data_adt.size(1)):
            a = str(adt_name[i],encoding="utf-8")
            adt_name_new.append(a)
        pd.DataFrame(sim_data_adt.cpu().numpy(), index = cell_name_sim, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_adt.csv'.format(mode,path))
        pd.DataFrame(real_data_adt.cpu().numpy(), index = cell_name_real, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_adt.csv'.format(mode,path))

        
    if mode == "SHAREseq":
        for i in range(sim_data_atac.size(1)):
            a = str(atac_name[i],encoding="utf-8")
            atac_name_new.append(a)           
        pd.DataFrame(sim_data_atac.cpu().numpy(), index = cell_name_sim, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_atac.csv'.format(mode,path))
        pd.DataFrame(real_data_atac.cpu().numpy(), index = cell_name_real, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_atac.csv'.format(mode,path))

        
    if mode == "TEAseq":
        for i in range(sim_data_adt.size(1)):
            a = str(adt_name[i],encoding="utf-8")
            adt_name_new.append(a)
        for i in range(sim_data_atac.size(1)):
            a = str(atac_name[i],encoding="utf-8")
            atac_name_new.append(a)            
        pd.DataFrame(sim_data_adt.cpu().numpy(), index = cell_name_sim, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_adt.csv'.format(mode,path))
        pd.DataFrame(real_data_adt.cpu().numpy(), index = cell_name_real, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_adt.csv'.format(mode,path))
        pd.DataFrame(sim_data_atac.cpu().numpy(), index = cell_name_sim, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_atac.csv'.format(mode,path))
        pd.DataFrame(real_data_atac.cpu().numpy(), index = cell_name_real, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_atac.csv'.format(mode,path))

    pd.DataFrame(sim_data_rna.cpu().numpy(), index = cell_name_sim, columns = rna_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_rna.csv'.format(mode,path))
    pd.DataFrame(real_data_rna.cpu().numpy(), index = cell_name_real, columns = rna_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_rna.csv'.format(mode,path))
    pd.DataFrame(sim_label_new,  index = cell_name_sim, columns = [ "label"]).to_csv( '../output/simulation_result/{}/{}/sim_label.csv'.format(mode,path))
    pd.DataFrame(real_label_new,  index = cell_name_real, columns = [ "label"]).to_csv( '../output/simulation_result/{}/{}/real_label.csv'.format(mode,path))

    print("finish simulation")

if (args.simulation == True) and (args.simulation_ct=="-1"):
    print("simulate all cells")
    checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    if not os.path.exists('../output/simulation_result/{}/{}/'.format(mode,path)):
        os.makedirs('../output/simulation_result/{}/{}/'.format(mode,path))        



################################################
################################################
################################################
    sim_data, temp1, temp2, temp3 = model(data)
    
    sim_label=label
    real_label=label
    
    sim_data_rna = sim_data[:, 0:nfeatures_rna]
    real_data_rna = data[:, 0:nfeatures_rna]   
    if mode == "CITEseq":
        sim_data_adt = sim_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
        real_data_adt = data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
    elif mode == "SHAREseq":
        sim_data_atac = sim_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_atac)]
        real_data_atac = data[:, nfeatures_rna:(nfeatures_rna+nfeatures_atac)]
    elif mode == "TEAseq":
        sim_data_adt = sim_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
        sim_data_atac = sim_data[:, (nfeatures_rna+nfeatures_adt):(nfeatures_rna+nfeatures_adt+nfeatures_atac)]
        real_data_adt = data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
        real_data_atac = data[:, (nfeatures_rna+nfeatures_adt):(nfeatures_rna+nfeatures_adt+nfeatures_atac)]
        
    rna_name_new = []
    adt_name_new = []
    atac_name_new = []
    
    b_list = range(0, real_data_rna.size(0))
    cell_name_real = ['cell_{}'.format(b) for b in b_list]
    b_list = range(0, sim_data_rna.size(0))
    cell_name_sim = ['cell_{}'.format(b) for b in b_list]
    sim_label_new = []
    real_label_new = []
    for j in range(sim_data_rna.size(0)):
        sim_label_new.append(transform_real_label[sim_label[j]])
    for j in range(real_data_rna.size(0)):    
        real_label_new.append(transform_real_label[real_label[j]])
    for i in range(sim_data_rna.size(1)):
        a = str(rna_name[i],encoding="utf-8")
        rna_name_new.append(a)           
        
    if mode == "CITEseq":
        for i in range(sim_data_adt.size(1)):
            a = str(adt_name[i],encoding="utf-8")
            adt_name_new.append(a)
        pd.DataFrame(sim_data_adt.cpu().detach().numpy(), index = cell_name_sim, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_adt.csv'.format(mode,path))
        pd.DataFrame(real_data_adt.cpu().detach().numpy(), index = cell_name_real, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_adt.csv'.format(mode,path))

        
    if mode == "SHAREseq":
        for i in range(sim_data_atac.size(1)):
            a = str(atac_name[i],encoding="utf-8")
            atac_name_new.append(a)           
        pd.DataFrame(sim_data_atac.cpu().detach().numpy(), index = cell_name_sim, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_atac.csv'.format(mode,path))
        pd.DataFrame(real_data_atac.cpu().detach().numpy(), index = cell_name_real, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_atac.csv'.format(mode,path))

        
    if mode == "TEAseq":
        for i in range(sim_data_adt.size(1)):
            a = str(adt_name[i],encoding="utf-8")
            adt_name_new.append(a)
        for i in range(sim_data_atac.size(1)):
            a = str(atac_name[i],encoding="utf-8")
            atac_name_new.append(a)            
        pd.DataFrame(sim_data_adt.cpu().detach().numpy(), index = cell_name_sim, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_adt.csv'.format(mode,path))
        pd.DataFrame(real_data_adt.cpu().detach().numpy(), index = cell_name_real, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_adt.csv'.format(mode,path))
        pd.DataFrame(sim_data_atac.cpu().detach().numpy(), index = cell_name_sim, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_atac.csv'.format(mode,path))
        pd.DataFrame(real_data_atac.cpu().detach().numpy(), index = cell_name_real, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_atac.csv'.format(mode,path))

    pd.DataFrame(sim_data_rna.cpu().detach().numpy(), index = cell_name_sim, columns = rna_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_rna.csv'.format(mode,path))
    pd.DataFrame(real_data_rna.cpu().detach().numpy(), index = cell_name_real, columns = rna_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_rna.csv'.format(mode,path))
    pd.DataFrame(sim_label_new,  index = cell_name_sim, columns = [ "label"]).to_csv( '../output/simulation_result/{}/{}/sim_label.csv'.format(mode,path))
    pd.DataFrame(real_label_new,  index = cell_name_real, columns = [ "label"]).to_csv( '../output/simulation_result/{}/{}/real_label.csv'.format(mode,path))

    print("finish simulation")
    
        
if args.dim_reduce == True:
    checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    simulated_data_ls, data_ls, label_ls = get_encodings(model,dl, args.device)
    simulated_data_ls[simulated_data_ls>torch.max(data)]=torch.max(data_ls)
    simulated_data_ls[simulated_data_ls<torch.min(data)]=torch.min(data_ls)
    simulated_data_ls[torch.isnan(simulated_data_ls)]=torch.max(data_ls)
    if not os.path.exists('../output/dim_reduce/{}/{}/'.format(mode,path)):
        os.makedirs('../output/dim_reduce/{}/{}/'.format(mode,path))
    b_list = range(0, simulated_data_ls.size(1))
    feature_index = ['feature_{}'.format(b) for b in b_list]   
    b_list = range(0, data.size(0))
    cell_name_real = ['cell_{}'.format(b) for b in b_list]  
    
    #if args.cty!="NULL":
    #    real_label_new = []
    #    for j in range(data.size(0)):
    #        real_label_new.append(transform_real_label[label[j]])
        #pd.DataFrame(real_label_new, index = cell_name_real, columns = [ "label"]).to_csv('../output/dim_reduce/{}/{}/latent_space_label.csv'.format(mode,path))
            
    #pd.DataFrame(simulated_data_ls.cpu().numpy(), index = cell_name_real, columns = feature_index).to_csv( '../output/dim_reduce/{}/{}/latent_space.csv'.format(mode,path))
    
    result = simulated_data_ls
    print(result.shape)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("create path")
    else:
        print("the path exits")

    file = h5py.File(args.save_path+"/embedding{}.h5".format(args.index), 'w')
    file.create_dataset('data', data=np.transpose(result.cpu().numpy()))
    file.close()
    
    print("finish dimension reduction")

if args.fs == True:
    rna_name_new = []
    adt_name_new = []
    atac_name_new = []
    for i in range(nfeatures_rna):
        a = str(rna_name[i],encoding="utf-8")
        rna_name_new.append(a)
    if mode == "CITEseq":
        for i in range(nfeatures_adt):
            a = str(adt_name[i],encoding="utf-8")
            adt_name_new.append(a)
        features = rna_name_new + adt_name_new
    if mode == "SHAREseq":
        for i in range(nfeatures_atac):
            a = str(atac_name[i],encoding="utf-8")
            atac_name_new.append(a)
        features = rna_name_new + atac_name_new
    if mode == "TEAseq":
        for i in range(nfeatures_adt):
            a = str(adt_name[i],encoding="utf-8")
            adt_name_new.append(a)
        for i in range(nfeatures_atac):
            a = str(atac_name[i],encoding="utf-8")
            atac_name_new.append(a)
        features = rna_name_new + adt_name_new + atac_name_new
        
    checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        
    classify_model = nn.Sequential(*list(model.children()))[0:2]
    
    if args.fs_method == "Saliency":
        deconv = Saliency(classify_model)
    else:
        deconv = IntegratedGradients(classify_model)
        
    for i in range(classify_dim):
        train_index_fs= torch.where(label==i)
        train_index_fs = [t.cpu().numpy() for t in train_index_fs]
        train_index_fs = np.array(train_index_fs)
        train_data_each_celltype_fs = data[train_index_fs,:].reshape(-1,feature_num)
    
        attribution = torch.zeros(1,feature_num)
        for j in range(train_data_each_celltype_fs.size(0)-1):
            attribution = attribution.to(device)+  torch.abs(deconv.attribute(train_data_each_celltype_fs[j:j+1,:], target=i))
        attribution_mean = torch.mean(attribution,dim=0)

        if not os.path.exists(save_fs_eachcell):
            os.makedirs(save_fs_eachcell)   
        pd.DataFrame(attribution_mean.cpu().numpy(), index = features, columns = [ "importance score"]).to_csv(save_fs_eachcell+"/fs."+"celltype_"+transform_real_label[i]+".csv")
    print("finish feature selection")
    

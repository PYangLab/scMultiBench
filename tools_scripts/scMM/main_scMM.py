import os
import sys
import h5py
import time
import torch
import models
import argparse
import datetime
import objectives
import numpy as np
import pandas as pd
from torch import optim
from pathlib import Path
import scipy.sparse as sp
from tempfile import mkdtemp
from scipy.io import mmwrite
from collections import defaultdict
from torch.utils.data import  DataLoader
from scipy.sparse import coo_matrix, csr_matrix
from datasets import RNA_Dataset, ATAC_Dataset
from utils import Timer, save_vars, EarlyStopping

parser = argparse.ArgumentParser(description='scMM')
parser.add_argument('--experiment', type=str, default='test', metavar='E',
                    help='experiment name')
parser.add_argument('--model', type=str, default='rna_protein', metavar='M',
                    help='model name (default: mnist_svhn)')
parser.add_argument('--obj', type=str, default='m_elbo_naive_warmup', metavar='O',
                    help='objective to use (default: elbo)')
parser.add_argument('--llik_scaling', type=float, default=1.,
                    help='likelihood scaling for cub images/svhn modality when running in'
                         'multimodal setting, set as 0 to use default value')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size for data (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='L',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--latent_dim', type=int, default=10, metavar='L',
                    help='latent dimensionality (default: 20)')
parser.add_argument('--num_hidden_layers', type=int, default=1, metavar='H',
                    help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--r_hidden_dim', type=int, default=100,
                    help='number of hidden units in enc/dec for gene')
parser.add_argument('--p_hidden_dim', type=int, default=20,
                    help='number of hidden units in enc/dec for protein/peak')
parser.add_argument('--pre_trained', type=str, default="",
                    help='path to pre-trained model (train from scratch if empty)')
parser.add_argument('--learn_prior', action='store_true', default=False,
                    help='learn model prior parameters')
parser.add_argument('--analytics', action='store_true', default=True,
                    help='disable plotting analytics')
parser.add_argument('--print_freq', type=int, default=0, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dataset_path', type=str, default="")
parser.add_argument('--path1', type=str, nargs='+', default=["./../../../../../../../project_mc3/chunlei/data/h5_data/CITEseq_PBMC/step1_train_rna.h5"])
parser.add_argument('--path2', type=str, nargs='+', default=["./../../../../../../../project_mc3/chunlei/data/h5_data/CITEseq_PBMC/step1_train_adt.h5"])
parser.add_argument('--save_path', type=str, default="")
parser.add_argument('--r_dim', type=int, default=1)
parser.add_argument('--p_dim', type=int, default=1)
parser.add_argument('--deterministic_warmup', type=int, default=50, metavar='W',
                    help='deterministic warmup')
args = parser.parse_args()

# The scMM script for vertical/cross integration requires one/multiple matched RNA+ADT or RNA+ATAC data as input. The output is a joint embedding (dimensionality reduction).
# run commond for scMM (RNA+ADT)
# python main_scMM.py --path1 "../../data/dataset_final/D3/rna.h5" --path2 "../../data/dataset_final/D3/adt.h5"  --save_path "../../result/embedding/D3/" --model rna_protein
# run commond for scMM (RNA+ATAC)
# python main_scMM.py --path1 "../../data/dataset_final/D15/rna.h5" --path2 "../../data/dataset_final/D15/atac.h5"  --save_path "../../result/embedding/D15/" --model rna_atac
# run commond for scMM (multiple RNA+ADT)
# python main_scMM.py --path1 "../../data/dataset_final/D51/rna1.h5" "../../data/dataset_final/D51/rna2.h5" --path2 "../../data/dataset_final/D51/adt1.h5"  "../../data/dataset_final/D51/adt2.h5" --save_path "../../result/embedding/cross integration/D51/scMM" --model rna_protein
# run commond for scMM (multiple RNA+ATAC)
# python main_scMM.py --path1 "../../data/dataset_final/SD18/rna1.h5" "../../data/dataset_final/SD18/rna2.h5" --path2 "../../data/dataset_final/SD18/atac1.h5" "../../data/dataset_final/SD18/atac2.h5"  --save_path "../../result/embedding/cross integration/SD18/scMM/" --model rna_atac


# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# set up run path
runId = datetime.datetime.now().isoformat()
experiment_dir = Path('../experiments/' + args.experiment)
experiment_dir.mkdir(parents=True, exist_ok=True)
runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
print(runPath)

#Data
dataset_path = args.dataset_path
if args.model == 'rna_atac':
    modal = 'ATAC-seq'
elif args.model == 'rna_protein':
    modal = 'CITE-seq'

rna_path = dataset_path + '/RNA-seq'
modal_path = dataset_path + '/{}'.format(modal)

####################################################################################
####################################################################################
####################################################################################

rna_paths =  args.path1 #
modal_paths = args.path2  #

rna_path_list = []
num=0
for rna_path in rna_paths:
    dir_path = os.path.dirname(rna_path)
    file_name = "/scMM_data/rna{}/".format(num)
    txt_path = dir_path + file_name
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    with h5py.File(rna_path, "r") as f:
        rna_X = np.mat(np.array(f['matrix/data']))
        rna_barcode = np.mat(np.array(f['matrix/barcodes']).transpose())
        rna_feature = np.mat(np.array(f['matrix/features']).transpose())
    sparse_rna = coo_matrix(rna_X)
    with open(txt_path+"count.mtx", 'w') as f:
        f.write('%%MatrixMarket matrix coordinate integer general\n')
        f.write('{} {} {}\n'.format(sparse_rna.shape[0], sparse_rna.shape[1], sparse_rna.nnz))
        for i, j, v in zip(sparse_rna.row + 1, sparse_rna.col + 1, sparse_rna.data):
            f.write('{} {} {}\n'.format(i, j, int(v)))
    np.savetxt(os.path.join(txt_path, "barcode.txt"), rna_barcode.transpose(), fmt='%s', delimiter='\n')
    np.savetxt(os.path.join(txt_path, "gene.txt"), rna_feature.transpose(), fmt='%s', delimiter='\n')
    rna_path = txt_path
    rna_path_list.append(txt_path)
    num = num+1

num=0
modal_path_list = []
for modal_path in modal_paths:
    dir_path = os.path.dirname(modal_path)
    file_name = "/scMM_data/another_modality{}/".format(num)
    txt_path = dir_path + file_name
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    with h5py.File(modal_path, "r") as f:
        adt_X = np.mat(np.array(f['matrix/data']))
        adt_barcode = np.mat(np.array(f['matrix/barcodes']).transpose())
        adt_feature = np.mat(np.array(f['matrix/features']).transpose())
    sparse_adt = coo_matrix(adt_X)
    with open(txt_path+"count.mtx", 'w') as f:
        f.write('%%MatrixMarket matrix coordinate integer general\n')
        f.write('{} {} {}\n'.format(sparse_adt.shape[0], sparse_adt.shape[1], sparse_adt.nnz))
        for i, j, v in zip(sparse_adt.row + 1, sparse_adt.col + 1, sparse_adt.data):
            f.write('{} {} {}\n'.format(i, j, int(v)))
    np.savetxt(os.path.join(txt_path, "barcode.txt"), adt_barcode.transpose(), fmt='%s', delimiter='\n')
    np.savetxt(os.path.join(txt_path, "protein.txt"), adt_feature.transpose(), fmt='%s', delimiter='\n')
    modal_path = txt_path
    modal_path_list.append(txt_path)
    num = num+1

####################################################################################
r_dataset_list = []
print(rna_path_list)
for rna_path in rna_path_list:
    r_dataset = RNA_Dataset(rna_path)
    args.r_dim = r_dataset.data.shape[1]
    r_dataset_list.append(r_dataset)
  
modal_dataset_list = []
for modal_path in modal_path_list:
    modal_dataset = ATAC_Dataset(modal_path) if args.model == 'rna_atac' else RNA_Dataset(modal_path)
    args.p_dim = modal_dataset.data.shape[1]
    modal_dataset_list.append(modal_dataset)

train_dataset = [r_dataset_list[0], modal_dataset_list[0]]
test_dataset_list = []
for i in range(len(modal_paths)):
    if i >= 1:
        test_dataset_list.append([r_dataset_list[i], modal_dataset_list[i]])
        
######################################################################################

# load model
modelC = getattr(models, 'VAE_{}'.format(args.model))
print(args)

model = modelC(args).to(device)
torch.save(args,runPath+'/args.rar')

#Dataloader
train_loader = model.getDataLoaders(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, device=device)
test_loader_list = []
for test_dataset in test_dataset_list:
    test_loader_list.append(model.getDataLoaders(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device))

# preparation for training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=args.lr, amsgrad=True)
objective = getattr(objectives,args.obj)
s_objective = getattr(objectives,args.obj)

def train(epoch, agg, W):
    model.train()
    b_loss = 0
    for i, dataT in enumerate(train_loader):
        beta = (epoch - 1) / W  if epoch <= W else 1
        if dataT[0].size()[0] == 1:
            continue
        data = [d.to(device) for d in dataT] #multimodal
        optimizer.zero_grad()
        loss = -objective(model, data, beta)
        loss.backward()
        optimizer.step()
        b_loss += loss.item()
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
    agg['train_loss'].append(b_loss / len(train_loader.dataset))
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))
    return b_loss

def test(epoch, agg, W, test_loader):
    model.eval()
    b_loss = 0
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            beta = (epoch - 1) / W  if epoch <= W else 1
            if dataT[0].size()[0] == 1:
                continue
            data = [d.to(device) for d in dataT]
            loss = -s_objective(model, data, beta)
            b_loss += loss.item()
    agg['test_loss'].append(b_loss / len(test_loader.dataset))
    print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))

if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=10, verbose=True)
        W = args.deterministic_warmup
        start_early_stop = W
        for epoch in range(1, args.epochs + 1):
            b_loss = train(epoch, agg, W)
            if torch.isnan(torch.tensor([b_loss])):
                break

    if args.analytics:
        def get_latent(dataloader, train_test, runPath):
            model.eval()
            with torch.no_grad():
                if args.model == 'rna_atac':
                    modal = ['rna', 'atac']
                elif args.model == 'rna_protein':
                    modal = ['rna', 'protein']
                pred = []
                for i, dataT in enumerate(dataloader):
                    data = [d.to(device) for d in dataT]
                    lats = model.latents(data, sampling=False)
                    if i == 0:
                        pred = lats
                    else:
                        for m,lat in enumerate(lats):
                            pred[m] = torch.cat([pred[m], lat], dim=0)
            
                for m,lat in enumerate(pred):
                    lat = lat.cpu().detach().numpy()
                    lat = pd.DataFrame(lat)
                    lat.to_csv('{}/lat_{}_{}.csv'.format(runPath, train_test, modal[m]))
                mean_lats = sum(pred)/len(pred)
                mean_lats = mean_lats.cpu().detach().numpy()
                mean_lats = pd.DataFrame(mean_lats)
                mean_lats.to_csv('{}/lat_{}_mean.csv'.format(runPath,train_test))
                return mean_lats

        def predict(dataloader, train_test, runPath):
            model.eval()
            with torch.no_grad():
                uni, cross = [], []
                for i, dataT in enumerate(dataloader):
                    data = [d.to(device) for d in dataT]
                    recons_mat = model.reconstruct_sample(data)
                    for e, recons_list in enumerate(recons_mat):
                        for d, recon in enumerate(recons_list):
                            if e == d:
                                recon = recon.squeeze(0).cpu().detach().numpy()
                                recon = pd.DataFrame(recon)
                                recon = sp.csr_matrix(recon)
                                if i == 0:
                                    uni.append(recon)
                                else:
                                    uni[e] = sp.vstack((uni[e], recon), format='csr')
                            if e != d:
                                recon = recon.squeeze(0).cpu().detach().numpy()
                                recon = pd.DataFrame(recon)
                                recon = sp.csr_matrix(recon)
                                if i == 0:
                                    cross.append(recon)
                                else:
                                    cross[e] = sp.vstack((cross[e], recon), format='csr')
                return uni, cross

        train_loader = model.getDataLoaders(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)
        test_loader_list = []
        for test_dataset in test_dataset_list:
            test_loader_list.append(model.getDataLoaders(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device))


        ################### save the embeddings for train and test data #################
        train_result = get_latent(train_loader, 'train', runPath)
        #train_result = np.transpose(train_result)
        test_result_list = []
        for test_loader in test_loader_list:
            test_result = get_latent(test_loader, 'test', runPath)
            #test_result = np.transpose(test_result)
            test_result_list.append(test_result)

        if test_result_list!=[]:
            test_result = np.concatenate(test_result_list,0)
            result = np.concatenate([train_result, test_result],0)
        else:
            result = train_result
        print(result.shape)
        
        #uni, cross = predict(train_loader, 'test', runPath)
                
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
            print("create path")
        else:
            print("the path exits")
        file = h5py.File(args.save_path+"/embedding.h5", 'w')
        file.create_dataset('data', data=result)
        file.close()
        

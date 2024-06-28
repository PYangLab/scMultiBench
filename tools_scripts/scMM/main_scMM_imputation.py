import argparse
import datetime
import sys
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
import torch.distributions as dist
from torch.nn.utils import clip_grad_norm_
from torch import optim
from torch.utils.data import Subset, DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset

import math
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix

import models
import objectives
from utils import Logger, Timer, save_model, save_vars, unpack_data, EarlyStopping, Constants, log_mean_exp, is_multidata, kl_divergence
#from datasets import RNA_Dataset, ATAC_Dataset

import numpy as np
import torch
import torch.distributions as dist
from torch.nn.utils import clip_grad_norm_
from torch import optim
from torch.utils.data import Subset, DataLoader
from torchnet.dataset import TensorDataset, ResampleDataset

import time 
import os 
import scipy
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import Dataset
import h5py

#python test.py --data_path "/home/sharonl/imputation_dataset/dataset43/data1" --train_fids '1' --impute_fids '2' --save_path './dataset43' --experiment 'rna_protein' --model 'rna_protein' --obj 'm_elbo_naive_warmup' --batch_size 32 --epochs 50 --deterministic_warmup 25 --lr 1e-4 --latent_dim 10 --num_hidden_layers 2 --r_hidden_dim 100 --p_hidden_dim 20 --dataset_path '../data/BMNC' --learn_prior
#python test.py --data_path "/home/sharonl/imputation_dataset/dataset43/data1" --train_fids '1' --impute_fids '2' --save_path './dataset43' --experiment 'rna_protein' --model 'rna_protein' --obj 'm_elbo_naive_warmup' --batch_size 32 --epochs 1 --deterministic_warmup 25 --lr 1e-4 --latent_dim 10 --num_hidden_layers 2 --r_hidden_dim 100 --p_hidden_dim 20 --dataset_path '../data/BMNC' --learn_prior

#python test.py --data_path "/home/sharonl/imputation_dataset/dataset44/data1" --train_fids '2' --impute_fids '3' --save_path './dataset44' --experiment 'rna_protein' --model 'rna_protein' --obj 'm_elbo_naive_warmup' --batch_size 32 --epochs 50 --deterministic_warmup 25 --lr 1e-4 --latent_dim 10 --num_hidden_layers 2 --r_hidden_dim 100 --p_hidden_dim 20 --dataset_path '../data/BMNC' --learn_prior
#python test.py --data_path "/home/sharonl/imputation_dataset/dataset45/data1" --train_fids '1' --impute_fids '2' --save_path './dataset45' --experiment 'rna_protein' --model 'rna_protein' --obj 'm_elbo_naive_warmup' --batch_size 32 --epochs 50 --deterministic_warmup 25 --lr 1e-4 --latent_dim 10 --num_hidden_layers 2 --r_hidden_dim 100 --p_hidden_dim 20 --dataset_path '../data/BMNC' --learn_prior
#python test.py --data_path "/home/sharonl/imputation_dataset/dataset46/data1" --train_fids '1' --impute_fids '2' --save_path './dataset46' --experiment 'rna_protein' --model 'rna_protein' --obj 'm_elbo_naive_warmup' --batch_size 32 --epochs 50 --deterministic_warmup 25 --lr 1e-4 --latent_dim 10 --num_hidden_layers 2 --r_hidden_dim 100 --p_hidden_dim 20 --dataset_path '../data/BMNC' --learn_prior
#python test.py --data_path "/home/sharonl/imputation_dataset/dataset47/data1" --train_fids '1' --impute_fids '2' --save_path './dataset47' --experiment 'rna_protein' --model 'rna_protein' --obj 'm_elbo_naive_warmup' --batch_size 32 --epochs 50 --deterministic_warmup 25 --lr 1e-4 --latent_dim 10 --num_hidden_layers 2 --r_hidden_dim 100 --p_hidden_dim 20 --dataset_path '../data/BMNC' --learn_prior


#python test.py --data_path "/home/sharonl/imputation_dataset/dataset48/data1" --train_fids '1' --impute_fids '2' --save_path './dataset48' --experiment 'rna_atac' --model 'rna_atac' --obj 'm_elbo_naive_warmup' --batch_size 32 --epochs 50 --deterministic_warmup 25 --lr 1e-4 --latent_dim 10 --num_hidden_layers 2 --r_hidden_dim 100 --p_hidden_dim 20 --dataset_path '../data/BMNC' --learn_prior
#python test.py --data_path "/home/sharonl/imputation_dataset/dataset49/data1" --train_fids '1' --impute_fids '2' --save_path './dataset49' --experiment 'rna_atac' --model 'rna_atac' --obj 'm_elbo_naive_warmup' --batch_size 32 --epochs 50 --deterministic_warmup 25 --lr 1e-4 --latent_dim 10 --num_hidden_layers 2 --r_hidden_dim 100 --p_hidden_dim 20 --dataset_path '../data/BMNC' --learn_prior

#python wrapper.py --data_path './dataset47' --mod 'adt' --save_path './results/dataset47'
#python wrapper.py --data_path './dataset49' --mod 'atac' --save_path './results/dataset49'

class RNA_Dataset(Dataset):
    """
    Single-cell RNA/ADT dataset
    """

    def __init__(self, path, transpose=False):
        
        self.data, self.genes, self.barcode = load_data(path, transpose)
        self.indices = None
        self.n_cells, self.n_peaks = self.data.shape
        self.shape = self.data.shape

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        if type(data) is not np.ndarray:
            data = data.toarray().squeeze()
        return torch.tensor(data)
    
    def info(self):
        print("\n===========================")
        print("Dataset Info")
        print('Cell number: {}\nGene number: {}'.format(self.n_cells, self.n_peaks))
        print('===========================\n')
        
        

def load_data(path, transpose=False):
    print("Loading  data ...")
    t0 = time.time()


    with h5py.File(path, "r") as f:
        #count = np.mat(np.array(f['matrix/data']).transpose())
        count = np.array(f['matrix/data']).transpose()
        peaks = [i.decode('utf-8') for i in f['matrix/features']]
        barcode = [i.decode('utf-8') for i in f['matrix/barcodes']]
        
    print('Original data contains {} cells x {} peaks'.format(*count.shape))
    assert (len(barcode), len(peaks)) == count.shape
    print("Finished loading takes {:.2f} min".format((time.time()-t0)/60))
    return count, peaks, barcode


class ATAC_Dataset(Dataset):
    """
    Single-cell ATAC dataset
    """

    def __init__(self, path, transpose=False):
        
        self.data, self.peaks, self.barcode = load_data(path, transpose)
        self.indices = None
        self.n_cells, self.n_peaks = self.data.shape
        self.shape = self.data.shape


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        if type(data) is not np.ndarray:
            data = data.toarray().squeeze()
        return torch.tensor(data)
    
    def info(self):
        print("\n===========================")
        print("Dataset Info")
        print('Cell number: {}\nPeak number: {}'.format(self.n_cells, self.n_peaks))
        print('===========================\n')


parser = argparse.ArgumentParser(description='scMM Hyperparameters')
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
parser.add_argument('--r_dim', type=int, default=1)
parser.add_argument('--p_dim', type=int, default=1)
parser.add_argument('--deterministic_warmup', type=int, default=50, metavar='W',
                    help='deterministic warmup')

parser.add_argument('--data_path', default='NULL', help='path to load the data')
parser.add_argument('--train_fids', metavar='trainid', nargs='+', default=[], help='file ids to train data1')
parser.add_argument('--impute_fids', metavar='imputeid', default='1', help='file ids to train data2')
parser.add_argument('--save_path', default='NULL', help='path to save the output data')
# args
args = parser.parse_args()

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
    model_fname = 'peak'
    modal = 'ATAC-seq'
elif args.model == 'rna_protein':
    modal = 'CITE-seq'
    model_fname = 'adt'




for trainid in args.train_fids:
    print("----preparing training data..")
    rna_h5 = os.path.join(args.data_path, 'reference', 'rna'+trainid+'.h5')
    adt_h5 = os.path.join(args.data_path, 'reference', model_fname+trainid+'.h5')
    print("->Loading "+rna_h5)
    print("->Loading "+adt_h5)

    r_dataset_train = RNA_Dataset(rna_h5) 
    args.r_dim = r_dataset_train.data.shape[1] 
    modal_dataset_train = ATAC_Dataset(adt_h5) if args.model == 'rna_atac' else RNA_Dataset(adt_h5)
    args.p_dim = modal_dataset_train.data.shape[1]
    print("RNA-seq (train) shape is " + str(r_dataset_train.data.shape))
    print("{} (train) shape is ".format(modal) + str(modal_dataset_train.data.shape)) 
    print()

    print("----preparing testing data..")
    rna_test_h5 = os.path.join(args.data_path, 'reference', 'rna'+args.impute_fids+'.h5')
    adt_test_h5 = os.path.join(args.data_path, 'gt', model_fname+args.impute_fids+'.h5')
    print("->Loading "+rna_test_h5)
    print("->Loading "+adt_test_h5)
    r_dataset_test = RNA_Dataset(rna_test_h5) 
    modal_dataset_test = ATAC_Dataset(adt_test_h5) if args.model == 'rna_atac' else RNA_Dataset(adt_test_h5)
    print("RNA-seq (test) shape is " + str(r_dataset_test.data.shape))
    print("{} (test) shape is ".format(modal) + str(modal_dataset_test.data.shape)) 
    print()

    break # we only have one datasets for training currently


#Split train test
train_dataset = [r_dataset_train, modal_dataset_train]
test_dataset = [r_dataset_test, modal_dataset_test]



# load args from disk if pretrained model path is given
pretrained_path = ""
if args.pre_trained:
    pretrained_path = args.pre_trained
    pretrain_args = args
    #pretrain_args.learn_prior = False

    #Load model
    modelC = getattr(models, 'VAE_{}'.format(pretrain_args.model))
    model = modelC(pretrain_args).to(device)
    print('Loading model {} from {}'.format(model.modelName, pretrained_path))
    model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
    model._pz_params = model._pz_params

else:
    # load model
    modelC = getattr(models, 'VAE_{}'.format(args.model))
    print(args)

    model = modelC(args).to(device)
    torch.save(args,runPath+'/args.rar')

#Dataloader
train_loader = model.getDataLoaders(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, device=device)
test_loader = model.getDataLoaders(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)

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

def test(epoch, agg, W):
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
            #test(epoch, agg, W)
            save_vars(agg, runPath + '/losses.rar')
            #if epoch > start_early_stop: 
            #    early_stopping(agg['test_loss'][-1], model, runPath)
            if early_stopping.early_stop:
                print('Early stopping')
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
                
                mmwrite('{}/pred_{}_r_r.mtx'.format(runPath, train_test), uni[0])
                mmwrite('{}/pred_{}_p_p.mtx'.format(runPath, train_test), uni[1])
                mmwrite('{}/pred_{}_r_p.mtx'.format(runPath, train_test), cross[0])
                mmwrite('{}/pred_{}_p_r.mtx'.format(runPath, train_test), cross[1])
        
        train_loader = model.getDataLoaders(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)
        test_loader = model.getDataLoaders(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, device=device)

        get_latent(train_loader, 'train', runPath)
        get_latent(test_loader, 'test', runPath)
        #print("@@@@@@@@@")
        predict(test_loader, 'test', runPath)
        
        
        print("---Saving data")
        file = h5py.File(args.save_path+"/imputed_result.h5",'w')
        #file.create_dataset("prediction", data=adt_imputed)

        file.create_dataset("groundtruth_rna_raw", data= r_dataset_test.data)
        file.create_dataset("groundtruth_other_raw", data= modal_dataset_test.data)
        model.traverse(runPath, device)


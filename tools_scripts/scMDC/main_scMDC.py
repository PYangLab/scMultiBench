
import os
import h5py
import time
import torch
import random
import numpy as np
from utils import *
import scanpy as sc
import time
import torch.nn as nn
import torch.nn.functional as F
from scMDC_batch import scMultiClusterBatch
from preprocess import read_dataset, normalize
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=27, type=int)
    parser.add_argument('--cutoff', default=0.5, type=float, help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='Normalized_filtered_BMNC_GSE128639_Seurat.h5')
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--gamma', default=.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--tau', default=1., type=float,
                        help='weight of clustering loss')
    parser.add_argument('--phi1', default=0.001, type=float,
                        help='coefficient of KL loss in pretraining stage')
    parser.add_argument('--phi2', default=0.001, type=float,
                        help='coefficient of KL loss in clustering stage')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--lr', default=1., type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/')
    parser.add_argument('--ae_weight_file', default='AE_weights_1.pth.tar')
    parser.add_argument('--resolution', default=0.2, type=float)
    parser.add_argument('--n_neighbors', default=30, type=int)
    parser.add_argument('--embedding_file', action='store_true', default=False)
    parser.add_argument('--prediction_file', action='store_true', default=False)
    parser.add_argument('-el','--encodeLayer', nargs='+', default=[256,64,32,16])
    parser.add_argument('-dl1','--decodeLayer1', nargs='+', default=[16,64,256])
    parser.add_argument('-dl2','--decodeLayer2', nargs='+', default=[16,20])
    parser.add_argument('--sigma1', default=2.5, type=float)
    parser.add_argument('--sigma2', default=1.5, type=float)
    parser.add_argument('--f1', default=1000, type=float, help='Number of mRNA after feature selection')
    parser.add_argument('--f2', default=2000, type=float, help='Number of ADT/ATAC after feature selection')
    parser.add_argument('--filter1', action='store_true', default=False, help='Do mRNA selection')
    parser.add_argument('--filter2', action='store_true', default=False, help='Do ADT/ATAC selection')
    parser.add_argument('--nbatch', default=2, type=int)
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--path1', metavar='DIR', default=[], nargs='+',  help='path to RNA')
    parser.add_argument('--path2', metavar='DIR', default=[], nargs='+',  help='path to ADT/ATAC')
    parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
    args = parser.parse_args()
    print(args)

# The scMDC script for vertical/cross integration requires one/multiple matched RNA+ADT or RNA+ATAC data as input. The output is a joint embedding (dimensionality reduction).
# run commond for scMDC (RNA+ADT)
# python main_scMDC.py --path1 "../../data/dataset_final/D3/rna.h5" --path2 "../../data/dataset_final/D3/adt.h5"  --save_path "../../result/embedding/D3/scMDC/" --nbatch 1  --n_clusters 26
# run commond for scMDC (RNA+ATAC)
# python main_scMDC.py --path1 "../../data/dataset_final/D15/rna.h5" --path2 "../../data/dataset_final/D15/atac.h5"  --save_path "../../result/embedding/D15/scMDC/" --nbatch 1  --n_clusters 13
# run commond for scMDC (multiple RNA+ADT)
# python main_scMDC.py --path1 "../../data/dataset_final/D51/rna1.h5" "../../data/dataset_final/D51/rna2.h5" --path2 "../../data/dataset_final/D51/adt1.h5"  "../../data/dataset_final/D51/adt2.h5" --save_path "../../result/embedding/cross integration/D51/scMDC" --nbatch 2 --n_clusters 17
# run commond for scMDC (multiple RNA+ATAC)
# python main_scMDC.py --path1 "../../data/dataset_final/SD18/rna1.h5" "../../data/dataset_final/SD18/rna2.h5" --path2 "../../data/dataset_final/SD18/atac1.h5" "../../data/dataset_final/SD18/atac2.h5"  --save_path "../../result/embedding/cross integration/SD18/scMDC/" --nbatch 2 --n_clusters 5


    begin_time = time.time()
    def data_loader(path):
        with h5py.File(path, "r") as f:
            X = np.mat(np.array(f['matrix/data']).transpose())
        return X


    def read_h5_file(file_paths):
        rna_path=file_paths['modality1_path']
        adt_path=file_paths['modality2_path']
        rna_list = []
        adt_list = []
        
        # read rna
        if rna_path is not None:
            for i in range(len(rna_path)):
                if rna_path[i] is None:
                    rna_list.append(None)
                else:
                    rna_list.append(data_loader(rna_path[i]))
        # read adt
        if adt_path is not None:
            for i in range(len(adt_path)):
                if adt_path[i] is None:
                    adt_list.append(None)
                else:
                    adt_list.append(data_loader(adt_path[i]))
                    
        batch = []
        for i in range(len(rna_list)):
            batch.append(np.ones((rna_list[i].shape[0], 1))  + i)
        batch = np.concatenate(batch,0)
        
        return np.concatenate(rna_list,0), np.concatenate(adt_list,0), batch
                    

    file_paths = {
        "modality1_path": 
            args.path1,
        "modality2_path": 
           args.path2
    }
    
    x1, x2, b = read_h5_file(file_paths)
    enc = OneHotEncoder()
    enc.fit(b.reshape(-1, 1))
    B = enc.transform(b.reshape(-1, 1)).toarray()

    #Gene filter
    if args.filter1:
        importantGenes = geneSelection(x1, n=args.f1, plot=False)
        x1 = x1[:, importantGenes]
    if args.filter2:
        importantGenes = geneSelection(x2, n=args.f2, plot=False)
        x2 = x2[:, importantGenes]

    # preprocessing scRNA-seq read counts matrix
    adata1 = sc.AnnData(x1)
    #adata1.obs['Group'] = y

    adata1 = read_dataset(adata1,
                     transpose=False,
                     test_split=False,
                     copy=True)

    print(adata1)
    adata1 = normalize(adata1,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    adata2 = sc.AnnData(x2)
    #adata2.obs['Group'] = y
    adata2 = read_dataset(adata2,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata2 = normalize(adata2,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    input_size1 = adata1.n_vars
    input_size2 = adata2.n_vars
    
    print(args)
    
    encodeLayer = list(map(int, args.encodeLayer))
    decodeLayer1 = list(map(int, args.decodeLayer1))
    decodeLayer2 = list(map(int, args.decodeLayer2))
    
    model = scMultiClusterBatch(input_dim1=input_size1, input_dim2=input_size2, n_batch = args.nbatch, tau=args.tau,
                        encodeLayer=encodeLayer, decodeLayer1=decodeLayer1, decodeLayer2=decodeLayer2,
                        activation='elu', sigma1=args.sigma1, sigma2=args.sigma2, gamma=args.gamma,
                        cutoff = args.cutoff, phi1=args.phi1, phi2=args.phi2, device=args.device).to(args.device)
    
    print(str(model))
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.ae_weights is None:
        model.pretrain_autoencoder(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors, 
                X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, B = B, batch_size=args.batch_size, 
                epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    
    latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device), torch.tensor(B).to(args.device), batch_size=args.batch_size)
    latent = latent.cpu().numpy()
    if args.n_clusters == -1:
       n_clusters = GetCluster(latent, res=args.resolution, n=args.n_neighbors)
    else:
       print("n_cluster is defined as " + str(args.n_clusters))
       n_clusters = args.n_clusters

    final_latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device), torch.tensor(B).to(args.device), batch_size=args.batch_size)
    final_latent = final_latent.cpu().numpy()
    result = final_latent
    end_time = time.time()
    all_time = end_time - begin_time
    result = np.transpose(result)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("create path")
    else:
        print("the path exits")
        
    file = h5py.File(args.save_path+"/embedding.h5", 'w')
    file.create_dataset('data', data=result)
    file.close()
    np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")

import h5py
import time
import scipy
import torch
import random
import sys, os
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from cobolt.model import Cobolt
from cobolt.utils import SingleData, MultiomicDataset


parser = argparse.ArgumentParser("Cobolt")
parser.add_argument('--path1', metavar='DIR', default="", help='path to RNA')
parser.add_argument('--path2', metavar='DIR', default="", help='path to RNA of multiome')
parser.add_argument('--path3', metavar='DIR', default="", help='path to ATAC of multiome')
parser.add_argument('--path4', metavar='DIR', default="", help='path to ATAC')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
parser.add_argument('--batch_size',  type=int,  default=128, help='path to save the output data')
parser.add_argument('--lr',  type=float,  default=0.005, help='path to save the output data')
args = parser.parse_args()

# The script of Cobolt for mosaic integration, [RNA, RNA+ATAC, ATAC] data type
# run commond for Cobolt
# python main_Cobolt.py --path1  "../../data/dataset_final/D45/rna1.h5"  --path2 "../../data/dataset_final/D45/rna2.h5" --path3 "../../data/dataset_final/D45/atac2.h5"  --path4 "../../data/dataset_final/D45/atac3.h5" --save_path "../../result/embedding/mosaic integration/D45/Cobolt/"

def h5_to_matrix(path):
    with h5py.File(path, "r") as f:
        X = (np.array(f['matrix/data']).transpose())
        X = sparse.csr_matrix(X)

        barcodes = f['matrix/barcodes'][:]
        features = f['matrix/features'][:]
        
        barcodes = np.array(barcodes)
        features = np.array(features)
        barcodes = np.array([ x.decode('utf-8') for x in barcodes])
        features = np.array([ x.decode('utf-8') for x in features])
    return X, barcodes, features

count, barcode, feature = h5_to_matrix(args.path2)
single_data_rna_2 = SingleData("GeneExpr", "snare", feature, count, barcode)

count, barcode, feature = h5_to_matrix(args.path3)
single_data_atac_2 = SingleData("ChromAccess", "snare", feature, count, barcode)

count, barcode, feature = h5_to_matrix(args.path1)
single_data_rna_1 = SingleData("GeneExpr", "mrna", feature, count, barcode)

count, barcode, feature = h5_to_matrix(args.path4)
single_data_atac_3 = SingleData("ChromAccess", "matac", feature, count, barcode)

single_data_rna_2.filter_features(upper_quantile=0.99, lower_quantile=0.7)
single_data_atac_2.filter_features(upper_quantile=0.99, lower_quantile=0.7)
single_data_atac_3.filter_features(upper_quantile=0.99, lower_quantile=0.7)
single_data_rna_1.filter_features(upper_quantile=0.99, lower_quantile=0.7)

multi_dt = MultiomicDataset.from_singledata(single_data_rna_2, single_data_atac_2, single_data_atac_3, single_data_rna_1)
model = Cobolt(dataset=multi_dt, lr=args.lr, n_latent=10, batch_size=args.batch_size)
model.train(num_epochs=100)
model.calc_all_latent()
latent = model.get_all_latent()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")

file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=np.transpose(latent[0]))
file.close()

latent_bytes = np.array(latent[1], dtype='S')
file = h5py.File(args.save_path+"/barcode.h5", 'w')
file.create_dataset('data', data=latent_bytes)
file.close()

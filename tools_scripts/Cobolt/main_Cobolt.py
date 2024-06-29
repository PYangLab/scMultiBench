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

random.seed(42)
parser = argparse.ArgumentParser("Cobolt")
parser.add_argument('--path1', metavar='DIR', default="", help='path to train data1')
parser.add_argument('--path2', metavar='DIR', default="", help='path to train data2')
parser.add_argument('--path3', metavar='DIR', default="", help='path to train data3')
parser.add_argument('--path4', metavar='DIR', default="", help='path to train data3')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
args = parser.parse_args()
begin_time = time.time()

class SingleData(object):
    def __init__(self,
                feature_name: str,
                dataset_name: str,
                feature: np.ndarray,
                count: sparse.csr.csr_matrix,
                barcode: np.ndarray):
        self.feature_name = feature_name
        self.dataset_name = dataset_name
        unique_feature, feature_idx = np.unique(feature, return_index=True)
        if len(feature) != len(unique_feature):
            print("Removing duplicated features.")
            feature = unique_feature
            count = count[:, feature_idx]
        self.feature = feature
        self.barcode = np.array([dataset_name + "~" + x.decode('utf-8') for x in barcode])
        self.count = count
        self.is_valid()


    @classmethod
    def from_h5(cls, path: str, feature_name: str, dataset_name: str):
        with h5py.File(path, 'r') as f:
            matrix_group = f['matrix']

            # Extract barcodes, features, and count matrix
            barcodes = matrix_group['barcodes'][:]
            features = [x.decode('utf-8') for x in matrix_group['features'][:]]
            data = matrix_group['data'][:].reshape(len(barcodes), len(features))
            count = sparse.csr_matrix(data)

        return cls(feature_name, dataset_name, np.array(features), count, barcodes)

    def __getitem__(self, items):
        x, y = items
        return SingleData(self.feature_name, self.dataset_name, self.feature[x],
                          self.count[x, y], self.barcode[y])

    def __str__(self):
        return "A SingleData object.\n" + \
               "Dataset name: {}. Feature name: {}.\n".format(
                   self.dataset_name, self.feature_name) + \
               "Number of features: {}. Number of cells {}.".format(
                   str(len(self.feature)), str(len(self.barcode)))

    def filter_features(self, min_count=10, min_cell=5, upper_quantile=1, lower_quantile=0):
        feature_count = np.sum(self.count, axis=0)
        feature_n = np.sum(self.count != 0, axis=0)
        bool_quality = np.array(
            (feature_n > min_cell) & (feature_count > min_count) &
            (feature_count >= np.quantile(feature_count, lower_quantile)) &
            (feature_count <= np.quantile(feature_count, upper_quantile))
        ).flatten()
        self.feature = self.feature[bool_quality]
        self.count = self.count[:, bool_quality]

    def filter_cells(self, min_count=10, min_feature=5, upper_quantile=1, lower_quantile=0):
        feature_count = np.sum(self.count, axis=1)
        feature_n = np.sum(self.count != 0, axis=1)
        bool_quality = np.array(
            (feature_n > min_feature) & (feature_count > min_count) &
            (feature_count >= np.quantile(feature_count, lower_quantile)) &
            (feature_count <= np.quantile(feature_count, upper_quantile))
        ).flatten()
        self.barcode = self.barcode[bool_quality]
        self.count = self.count[bool_quality, :]

    def filter_barcode(self, cells):
        bool_cells = np.isin(self.barcode, cells)
        self.count = self.count[bool_cells, :]
        self.barcode = self.barcode[bool_cells]

    def subset_features(self, feature):
        bool_features = np.isin(self.feature, feature)
        self.count = self.count[:, bool_features]
        self.feature = self.feature[bool_features]

    def rename_features(self, feature):
        unique_feature, feature_idx = np.unique(feature, return_index=True)
        if len(feature) != len(unique_feature):
            print("Removing duplicated features.")
            feature = unique_feature
            self.count = self.count[:, feature_idx]
        self.feature = np.array(feature)

    def get_data(self):
        return {self.feature_name: self.count}, {self.feature_name: self.feature}, self.barcode

    def get_dataset_name(self):
        return self.dataset_name

    def is_valid(self):
        if self.count.shape[0] != self.barcode.shape[0]:
            raise ValueError("The dimensions of the count matrix and the barcode array are not consistent.")
        if self.count.shape[1] != self.feature.shape[0]:
            raise ValueError("The dimensions of the count matrix and the barcode array are not consistent.")
            
def train_and_analyze_model(
    single_data_list,
    learning_rate=0.0005,
    n_latent=10,
    num_epochs=100,
    clustering_algo="leiden",
    clustering_resolution=0.5,
    scatter_reduc="UMAP",
    scatter_size=0.2,
):
    print(learning_rate)

    # Create a MultiomicDataset from the given single datasets
    multi_dt = MultiomicDataset.from_singledata(*single_data_list)
    print(multi_dt)

    # Initialize and train the model
    model = Cobolt(dataset=multi_dt, lr=learning_rate, n_latent=n_latent)
    model.train(num_epochs=num_epochs)

    # Calculate and retrieve latent variables
    model.calc_all_latent()
    latent = model.get_all_latent()

    return latent

single_data_rna_1 = SingleData.from_h5(args.path1, 'GeneExpr', 'batch1')
print(single_data_rna_1)
single_data_rna_2 = SingleData.from_h5(args.path2, 'GeneExpr', 'batch2')
print(single_data_rna_2)
single_data_atac_2 = SingleData.from_h5(args.path3,  'ChromAccess', 'batch2')
print(single_data_atac_2)
single_data_atac_3 = SingleData.from_h5(args.path4, 'ChromAccess', 'batch3')
print(single_data_atac_3)

single_data_rna_2.filter_features(upper_quantile=0.99, lower_quantile=0.7)
single_data_atac_2.filter_features(upper_quantile=0.99, lower_quantile=0.7)
single_data_atac_3.filter_features(upper_quantile=0.99, lower_quantile=0.7)
single_data_rna_1.filter_features(upper_quantile=0.99, lower_quantile=0.7)

result = train_and_analyze_model([single_data_rna_2, single_data_atac_2, single_data_atac_3, single_data_rna_1])
print(result[0].shape)

end_time = time.time()
all_time = end_time - begin_time
print(all_time)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path")
else:
    print("the path exits")
file = h5py.File(args.save_path+"/embedding.h5", 'w')
file.create_dataset('data', data=np.transpose(result[0]))
file.close()
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")

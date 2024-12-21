import os
import h5py
import time
import random
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from anndata import AnnData, read_h5ad
from sciPENN.sciPENN_API import sciPENN_API
from sciPENN.Preprocessing import preprocess

parser = argparse.ArgumentParser("sciPENN")
parser.add_argument('--data_path', default='NULL', help='path to load the data')
parser.add_argument('--train_fids', metavar='trainid', nargs='+', default=[], help='file ids to train data1')
parser.add_argument('--impute_fids', metavar='imputeid', default='1', help='file ids to train data2')
parser.add_argument('--save_path', default='NULL', help='path to save the output data')
args = parser.parse_args()
print(args)

# The script sciPENN is designed for mosaic integration (imputation). It can predict the missing ADT modality using a CITE-seq reference and an RNA query.
# run commond for sciPENN
# python main_sciPENN_imputation.py --data_path "../../data/dataset_final_imputation_hvg/D52/data1/" --train_fids '1' --impute_fids '2'   --save_path "../../result/imputation_filter/D52/data1/sciPENN/"

def data_loader(path, bid):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).transpose()
        feat = [i.decode('utf-8') for i in f['matrix/features']]
        cid = [i.decode('utf-8') for i in f['matrix/barcodes']]
    adata = AnnData(X=X)
    adata.obs['bid'] = str(bid)
    adata.var_names = feat
    adata.obs_names = cid
    return adata

begin_time = time.time()
gene_training_list = []
protein_training_list = []

print("----preparing training data..")
for trainid in args.train_fids:
	rna_h5 = os.path.join(args.data_path, 'reference', 'rna'+trainid+'.h5')
	adt_h5 = os.path.join(args.data_path, 'reference', 'adt'+trainid+'.h5')
	lb_csv = os.path.join(args.data_path, 'reference', 'cty'+trainid+'.csv')
	lbs = pd.read_csv(lb_csv)['x'].tolist()
	print("->Loading "+rna_h5)
	genes = data_loader(rna_h5, trainid)
	genes.obs['lbs'] = lbs
	gene_training_list.append(genes)
	print("->Loading "+adt_h5)
	proteins = data_loader(adt_h5, trainid)
	proteins.obs['lbs'] = lbs
	protein_training_list.append(proteins)

print("----preparing test data..")
rna_test_h5 = os.path.join(args.data_path, 'reference', 'rna'+args.impute_fids+'.h5')
adt_test_h5 = os.path.join(args.data_path, 'gt', 'adt'+args.impute_fids+'.h5')
lb_test_csv = os.path.join(args.data_path, 'reference', 'cty'+args.impute_fids+'.csv')
lbs_test = pd.read_csv(lb_test_csv)['x'].tolist()
print("->Loading "+rna_test_h5)
gene_testing = data_loader(rna_test_h5, args.impute_fids)
gene_testing.obs['lbs'] = lbs_test
print("->Loading "+adt_test_h5)
adt_testing = data_loader(adt_test_h5, args.impute_fids)
adt_testing.obs['lbs'] = lbs_test

print("---initializing sciPENN_API")
sciPENN = sciPENN_API(gene_trainsets = gene_training_list, protein_trainsets = protein_training_list, gene_test = gene_testing, train_batchkeys = ['bid' for j in range(len(gene_training_list))], test_batchkey = 'bid')
print("---training sciPENN")
sciPENN.train(quantiles = [0.1, 0.25, 0.75, 0.9], n_epochs = 10000, ES_max = 12, decay_max = 6,
             decay_step = 0.1, lr = 10**(-3), weights_dir = args.save_path, load = False)
print("---conducting imputation")
predicted_test = sciPENN.predict()
end_time = time.time()
all_time = end_time - begin_time
print(all_time)
np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")


print("---normalizing ground truth")
_, protein_gt, _, _, _, _ = preprocess(gene_trainsets=[gene_testing], protein_trainsets=[adt_testing], train_batchkeys=['bid'], min_cells = 0, min_genes = 0)
print(protein_gt.X)
print(protein_gt.X.shape)
print(protein_gt)

print("---Saving data")
file = h5py.File(args.save_path+"/imputed_result.h5",'w')
file.create_dataset("prediction", data=predicted_test.X)
file.create_dataset("groundtruth_raw", data= adt_testing.X)
file.create_dataset("groundtruth_norm", data= protein_gt.X)




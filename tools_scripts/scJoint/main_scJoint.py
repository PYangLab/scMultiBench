import h5py
import os
import argparse
import numpy as np
import time
import pandas as pd
import torch
import os
from datetime import datetime

from config import Config
from util.trainingprocess_stage1 import TrainingProcessStage1
from util.trainingprocess_stage3 import TrainingProcessStage3
from util.knn import KNN
import parser
import argparse
import h5py
import process_db
import pandas as pd

###
### this code is from https://github.com/rsinghlab/SCOT
begin_time = time.time()

parser = argparse.ArgumentParser("scJoint")
parser.add_argument('--threads', type=int, default=1, help='threads')
parser.add_argument('--use_cuda', type=bool, default= True, help='if use gpu or not')
parser.add_argument('--with_crossentorpy', type=bool, default= True, help='if use _crossentorpy or not')

parser.add_argument('--path1', metavar='DIR', nargs='*', default=[], help='path to data1')
parser.add_argument('--path2', metavar='DIR', nargs='*', default=[], help='path to data2')
parser.add_argument('--cty_path1', metavar='DIR', nargs='*', default=[], help='path to cty1')
parser.add_argument('--cty_path2', metavar='DIR', nargs='*', default=[], help='path to cty2')
parser.add_argument('--save_path', metavar='DIR', default='NULL', help='path to save the output data')
parser.add_argument('--checkpoint', metavar='DIRS', nargs='+',default=[], help='path to train cell type label')
parser.add_argument('--rna_protein_paths', metavar='DIRS', nargs='+',default=[], help='path to train cell type label')
parser.add_argument('--atac_protein_paths', metavar='DIRS', nargs='+',default=[], help='path to train cell type label')

##############  for training #################
parser.add_argument('--number_of_class', type=int, default=7, help='number of class')
parser.add_argument('--input_size', type=int, default=17668, help='int size')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr_stage1', type=float, default=0.01, help='batch size')
parser.add_argument('--lr_stage3', type=float, default=0.01, help='batch size')
parser.add_argument('--lr_decay_epoch', type=int, default=20, help='batch size')
parser.add_argument('--epochs_stage1', type=int, default=20, help='batch size')
parser.add_argument('--epochs_stage3', type=int, default=20, help='batch size')
parser.add_argument('--p', type=float, default=0.8, help='batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--embedding_size', type=int, default=64, help='batch size')
parser.add_argument('--center_weight', type=int, default=1, help='center_weight')
parser.add_argument('--seed', type=int, default=1, help='seed')
args = parser.parse_args()

if not args.use_cuda:
    args.device = torch.device('cpu')
else:
    args.device = torch.device('cuda:0')
        
args.rna_paths = args.path1
args.atac_paths = args.path2
args.rna_labels = args.cty_path1
args.atac_labels = args.cty_path2

def main():    
    ###### data preprocessing ######
    begin_time = time.time()
    rna_h5_files = args.rna_paths
    rna_label_files = args.rna_labels

    atac_h5_files = args.atac_paths
    atac_label_files = args.atac_labels
    
    h5 = h5py.File(rna_h5_files[0], "r")
    h5_data = h5['matrix/data']
    process_db.data_parsing(rna_h5_files, atac_h5_files)
    
    rna_label = pd.read_csv(rna_label_files[0], index_col = 0)
    rna_label
    print(rna_label.value_counts(sort = False))
    if (atac_label_files != []):
        atac_label = pd.read_csv(atac_label_files[0], index_col = 0)
        atac_label
        print(atac_label.value_counts(sort = False))
        atac_label_files_temp = atac_label_files
    else:
        atac_label = []
        atac_label_files_temp = []
    
    feature_num = h5_data.shape[0]
    num_cty = 50 #rna_label['x'].nunique()
    
    process_db.label_parsing(rna_label_files, atac_label_files_temp)

    args.rna_paths = [file.replace(".h5", ".npz") for file in rna_h5_files]
    args.atac_paths = [file.replace(".h5", ".npz") for file in atac_h5_files]
    args.rna_labels = [file.replace(".csv", ".txt") for file in rna_label_files]

    if (atac_label_files != []):
        args.atac_labels = [file.replace(".csv", ".txt") for file in atac_label_files]
    else:
        args.atac_labels =  []
    
    args.input_size = feature_num 
    args.number_of_class = num_cty
    # hardware constraint for speed test
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.manual_seed(args.seed)
    print('Start time: ', datetime.now().strftime('%H:%M:%S'))

    # stage1 training
    print('Training start [Stage1]')
    model_stage1= TrainingProcessStage1(args)    
    for epoch in range(args.epochs_stage1):
        print('Epoch:', epoch)
        model_stage1.train(epoch)
    
    print('Write embeddings')
    model_stage1.write_embeddings()
    print('Stage 1 finished: ', datetime.now().strftime('%H:%M:%S'))
    
    # KNN
    print('KNN')
    KNN(args, neighbors = 30, knn_rna_samples=20000)
    print('KNN finished: ', datetime.now().strftime('%H:%M:%S'))
    
    # stage3 training
    print('Training start [Stage3]')
    model_stage3 = TrainingProcessStage3(args)    
    for epoch in range(args.epochs_stage3):
       print('Epoch:', epoch)
       model_stage3.train(epoch)
        
    print('Write embeddings [Stage3]')
    model_stage3.write_embeddings()
    print('Stage 3 finished: ', datetime.now().strftime('%H:%M:%S'))
    
    # KNN
    print('KNN stage3')
    KNN(args, neighbors = 30, knn_rna_samples=20000)
    print('KNN finished: ', datetime.now().strftime('%H:%M:%S'))
    
    end_time = time.time()
    all_time = end_time - begin_time
    print(all_time)
    
    #########save###########
    rna_embeddings_list = []
    for i in args.path1:
        #rna_file_name = os.path.basename(args.path1)
        rna_file_name = os.path.basename(i)
        rna_name = rna_file_name.split('.')[0]
        rna_embeddings = np.loadtxt('./output/{}_embeddings.txt'.format(rna_name))
        rna_embeddings_list.append(rna_embeddings)
            
    atac_embeddings_list = []
    for i in args.path2:
        atac_file_name = os.path.basename(i)
        atac_name = atac_file_name.split('.')[0]
        atac_embeddings = np.loadtxt('./output/{}_embeddings.txt'.format(atac_name))
        atac_embeddings_list.append(atac_embeddings)

    rna_embeddings = np.concatenate(rna_embeddings_list,0)
    atac_embeddings = np.concatenate(atac_embeddings_list,0)
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("create path")
    else:
        print("the path exits")
    embedding = np.concatenate([rna_embeddings, atac_embeddings], 0)
    file = h5py.File(args.save_path+"/embedding.h5", 'w')
    file.create_dataset('data', data=embedding)
    file.close()
    np.savetxt(args.save_path+"/time.csv", [all_time], delimiter=",")
    
if __name__ == "__main__":
    main()


    

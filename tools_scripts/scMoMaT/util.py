import h5py
import torch
import scipy
import scmomat
import sys, os
import numpy as np
import pandas as pd  
import scipy.sparse as sp

def read_fs_label(label_paths):
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    all_label = []
    for label_path in label_paths:
        label_fs = pd.read_csv(label_path,header=None,index_col=False)  #
        label_fs = label_fs.iloc[1:(label_fs.shape[0]),1]
        label_fs = pd.Categorical(label_fs).codes
        label_fs = np.array(label_fs[:]).astype('int32')
        label_fs = torch.from_numpy(label_fs)
        label_fs = label_fs.type(LongTensor)
        all_label.append(label_fs)
    all_label = torch.cat(all_label,0)
    return all_label

def h5_to_matrix(path):
    data = h5py.File(path,"r")
    h5_data = data['matrix/data']
    sparse_data = scipy.sparse.csr_matrix(np.array(h5_data).transpose())
    data = np.array(sparse_data.todense())
    return data
    
def read_h5_data(rna_path=None, adt_path=None, atac_path=None, list_len=3):
    rna_list = []
    adt_list = []
    atac_list = []
    rna_feature_num = "None"
    adt_feature_num = "None"
    atac_feature_num = "None"
    
    # read rna
    if rna_path != "None":
        for i in range(len(rna_path)):
            if rna_path[i] == "None":
                rna_list.append(None)
            else:
                rna_feature_num = h5_to_matrix(rna_path[i]).shape[1]
                rna_list.append(scmomat.preprocess((h5_to_matrix(rna_path[i])), modality = "RNA"))
    
    # read adt
    if adt_path != "None":
        for i in range(len(adt_path)):
            if adt_path[i] == "None":
                adt_list.append(None)
            else:
                adt_feature_num = h5_to_matrix(adt_path[i]).shape[1]
                adt_list.append(scmomat.preprocess((h5_to_matrix(adt_path[i])), modality = "ADT"))
    
    # read atac
    if atac_path != "None":
        for i in range(len(atac_path)):
            if atac_path[i] == "None":
                atac_list.append(None)
            else:
                atac_feature_num = h5_to_matrix(atac_path[i]).shape[1]
                atac_list.append(scmomat.preprocess((h5_to_matrix(atac_path[i])), modality = "ATAC"))
    
    result = {"rna": rna_list, "adt": adt_list, "atac": atac_list}
    feature_num = [rna_feature_num, adt_feature_num, atac_feature_num]
    return result, feature_num


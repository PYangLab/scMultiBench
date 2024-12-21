import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from util import setup_seed, MyDataset, save_checkpoint,AverageMeter,accuracy,ToTensor,read_h5_data,read_fs_label

import random
import os
from torch.autograd import Variable
import parser
import argparse

parser = argparse.ArgumentParser("classification")
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--mode', type=int, default=12, help='random seed')
parser.add_argument('--lr', type=float, default=1e-2, help='init learning rate')
parser.add_argument('--reference', metavar='DIR', default="", help='path to train rna data')
parser.add_argument('--query', metavar='DIR', default="", help='path to train adt data')
parser.add_argument('--reference_cty', metavar='DIR', default="", help='path to train atac data')
parser.add_argument('--query_cty', metavar='DIR', default="", help='path to train cell type label')
parser.add_argument('--save_path', metavar='DIR', default="", help='path to train cell type label')
args = parser.parse_args()


setup_seed(1)
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor



data_path1 = args.reference
data_path2 = args.query
cty_path1 = args.reference_cty
cty_path2 = args.query_cty
    
rna_train_data_num,feature_num = read_h5_data(data_path1).size()
rna_test_data_num,feature_num = read_h5_data(data_path2).size()

rna_train_data = read_h5_data(data_path1)
rna_test_data= read_h5_data(data_path2)

rna_train_label = read_fs_label(cty_path1)  #
rna_test_label = read_fs_label(cty_path2)  #

classify_dim = int(torch.max(rna_train_label) + 1)

def real_label(label_path,classify_dim):
    output_v = []
    label = pd.read_csv(label_path,header=None,index_col=False)  #
    label_real = label.iloc[1:(label.shape[0]),1]
    label_num = read_fs_label(label_path)
    for i in range(classify_dim):
        temp = label_real[np.array(torch.where(label_num==i)[0][0].cpu()).astype('int32')+1]
        output_v.append(temp)
    return output_v
    
transform_real_label = real_label(cty_path1, classify_dim)
        
class MyDataset_fs(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        img, target = self.data[index,:], self.label[index]
        sample = {'data': img, 'label': target}
        return sample

    def __len__(self):
        return len(self.data)

train_transformed_dataset = MyDataset_fs(rna_train_data,
                                rna_train_label
                                    )
train_dataloader = DataLoader(train_transformed_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0,drop_last=False)

test_transformed_dataset = MyDataset_fs(rna_test_data,
                                rna_test_label
                                    )
test_dataloader = DataLoader(test_transformed_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0,drop_last=False)



class Net_fs(nn.Module):
    def __init__(self):
        super(Net_fs, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(feature_num, classify_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        return x

model_fs = Net_fs().cuda()
optimizer_fs = torch.optim.Adam(model_fs.parameters(), lr=args.lr)
criterion_fs = nn.CrossEntropyLoss()

top1 = AverageMeter('Acc@1', ':6.2f')
for epoch in range(args.epochs):
    for i, batch_sample in enumerate(train_dataloader):#(imgs, labels) in enumerat
        model_fs.train()
        rna_train_data = batch_sample['data']
        rna_train_label = batch_sample['label']
        
        # Configure input
        rna_train_data = Variable(rna_train_data.type(FloatTensor))
        rna_train_label = Variable(rna_train_label.type(LongTensor))

        #training
        optimizer_fs.zero_grad()
        rna_train_output = model_fs(rna_train_data)
        loss = criterion_fs(rna_train_output, rna_train_label)
        loss.backward()
        optimizer_fs.step()
        
        a = torch.max(nn.Softmax()(rna_train_output),1)
        pred1,  = accuracy(rna_train_output, rna_train_label, topk=(1, ))
        top1.update(pred1[0], 1)

predict_label = model_fs(rna_test_data)
predict_label = torch.max(nn.Softmax()(predict_label),1).indices
print(len(predict_label))
print(len(rna_test_label))
print(predict_label == rna_test_label)
print(torch.sum(predict_label == rna_test_label)/len(rna_test_label),"!!!!!!!")

real_predict_label = []
for j in range(predict_label.size(0)):
    real_predict_label.append(transform_real_label[predict_label[j]])

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
pd.DataFrame(real_predict_label).to_csv(args.save_path+"/predict"+".csv")

real_query_label = []
for j in range(predict_label.size(0)):
    real_query_label.append(transform_real_label[rna_test_label[j]])
pd.DataFrame(real_query_label).to_csv(args.save_path+"/query"+".csv")
 

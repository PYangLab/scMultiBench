import os
import torch
import random
import anndata
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from util import setup_seed, MyDataset, save_checkpoint,AverageMeter,accuracy,ToTensor,read_h5_data,read_fs_label

parser = argparse.ArgumentParser("classification")
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--mode', type=int, default=12, help='random seed')
parser.add_argument('--lr', type=float, default=1e-2, help='init learning rate')
parser.add_argument('--data_path1', metavar='DIR', default=["../../../dataset/classification/cross integration/dataset43/totalVI/data1.h5"], help='path to train data')
parser.add_argument('--data_path2', metavar='DIR', nargs='+', default=["../../../dataset/classification/cross integration/dataset43/totalVI/data2.h5"], help='path to test data')
parser.add_argument('--cty_path1', metavar='DIR', default="../../../dataset/classification/cross integration/dataset43/totalVI/cty1.csv", help='path to train labels')
parser.add_argument('--cty_path2', metavar='DIR', nargs='+',  default=["../../../dataset/classification/cross integration/dataset43/totalVI/cty2.csv"], help='path to test labels')
parser.add_argument('--save_path', metavar='DIR', default="../../../result/classification/cross integration/dataset43/totalVI/case1/", help='path to save results')
args = parser.parse_args()
print(args)

# this script is designed for unifled MLP classifier
# run script
# python main.py --data_path1 "./data1.h5" --data_path2 "./data2.h5" --cty_path1 "./cty1.csv" --cty_path2 "./cty2.csv" --save_path "./"

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def concat_h5_data(paths):
    if len(paths) == 1:
        data = read_h5_data(paths[0])
    else:
        datasets = [read_h5_data(path) for path in paths]
        data = torch.cat(datasets, dim=0)
    return data

def read_and_concat_csv_data(paths):
    if len(paths) == 1:
        label_fs_df = pd.read_csv(paths[0])
    else:
        dataframes = [pd.read_csv(path) for path in paths]
        label_fs_df = pd.concat(dataframes, axis=0, ignore_index=True)
    return label_fs_df


reference_label_df = pd.read_csv(args.cty_path1)
reference_cell_types = reference_label_df['x'].unique()
classify_dim = len(reference_cell_types)
print(f"Reference cell types: {reference_cell_types}")

train_label_codes = pd.Categorical(reference_label_df['x'], categories=reference_cell_types).codes
rna_train_label = torch.from_numpy(train_label_codes.astype('int32')).type(LongTensor)
print(f"Train label size: {rna_train_label.shape}")

rna_train_data = read_h5_data(args.data_path1)
print(f"Train data size: {rna_train_data.shape}")


test_label_df = read_and_concat_csv_data(args.cty_path2)
print(f"Original test label size: {test_label_df.shape}")

mask = test_label_df['x'].isin(reference_cell_types)
filtered_test_label_df = test_label_df.loc[mask].reset_index(drop=True)
indices_to_keep = test_label_df.index[mask].tolist()
print(f"Filtered test label size: {filtered_test_label_df.shape}")

rna_test_data = concat_h5_data(args.data_path2)
print(f"Original test data size: {rna_test_data.size()}")

rna_test_data = rna_test_data[indices_to_keep, :]
rna_test_data_num, feature_num = rna_test_data.size()
print(f"Filtered test data size: {rna_test_data_num} samples, {feature_num} features")

label_fs = pd.Categorical(filtered_test_label_df['x'], categories=reference_cell_types).codes
label_fs = np.array(label_fs).astype('int32')
rna_test_label = torch.from_numpy(label_fs).type(LongTensor)
print(f"Filtered test label tensor size: {rna_test_label.shape}")

class Net_fs(nn.Module):
    def __init__(self):
        super(Net_fs, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(feature_num, 128),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(128, classify_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

model_fs = Net_fs()
if cuda:
    model_fs = model_fs.cuda()
optimizer_fs = torch.optim.Adam(model_fs.parameters(), lr=args.lr)
criterion_fs = nn.CrossEntropyLoss()

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

train_transformed_dataset = MyDataset_fs(rna_train_data, rna_train_label)
train_dataloader = DataLoader(train_transformed_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0, drop_last=False)

test_transformed_dataset = MyDataset_fs(rna_test_data, rna_test_label)
test_dataloader = DataLoader(test_transformed_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, drop_last=False)

for epoch in range(args.epochs):
    model_fs.train()
    for i, batch_sample in enumerate(train_dataloader):
        rna_train_data_batch = batch_sample['data']
        rna_train_label_batch = batch_sample['label']
        
        # Configure input
        rna_train_data_batch = Variable(rna_train_data_batch.type(FloatTensor))
        rna_train_label_batch = Variable(rna_train_label_batch.type(LongTensor))

        # Training
        optimizer_fs.zero_grad()
        rna_train_output = model_fs(rna_train_data_batch)
        loss = criterion_fs(rna_train_output, rna_train_label_batch)
        loss.backward()
        optimizer_fs.step()
    
    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}")

model_fs.eval()
correct = 0
total = 0
all_predicted = []
all_true_labels = []
with torch.no_grad():
    for batch_sample in test_dataloader:
        rna_test_data_batch = batch_sample['data']
        rna_test_label_batch = batch_sample['label']
        rna_test_data_batch = Variable(rna_test_data_batch.type(FloatTensor))
        outputs = model_fs(rna_test_data_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += rna_test_label_batch.size(0)
        correct += (predicted.cpu() == rna_test_label_batch.cpu()).sum()
        

        all_predicted.extend(predicted.cpu().numpy())
        all_true_labels.extend(rna_test_label_batch.cpu().numpy())

print(f'Accuracy of the model on the test data: {100 * correct / total}%')
real_predict_label = [reference_cell_types[idx] for idx in all_predicted]
real_query_label = [reference_cell_types[idx] for idx in all_true_labels]

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
pd.DataFrame(real_predict_label).to_csv(os.path.join(args.save_path, "predict.csv"), index=False)
pd.DataFrame(real_query_label).to_csv(os.path.join(args.save_path, "query.csv"), index=False)

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:17:45 2020

@author: Yang Xu
"""
import gc
import numpy as np

import torch
import torch.nn.functional as F
#from contrastive_loss_pytorch import ContrastiveLoss
import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).cuda())
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, 
                                                           dtype=bool)).float().cuda())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        #z_i = F.normalize(emb_i, dim=1)
        #z_j = F.normalize(emb_j, dim=1)
        
        z_i = F.normalize(emb_i, dim=1,p=2)
        z_j = F.normalize(emb_j, dim=1,p=2)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
    
##-----------------------------------------------------------------------------
class SMILE(torch.nn.Module):
    def __init__(self,input_dim=2000,clf_out=10):
        super(SMILE, self).__init__()
        self.input_dim = input_dim
        self.clf_out = clf_out
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(128, self.clf_out),
            torch.nn.Softmax(dim=1))
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(128, 32))
        
    def forward(self, x):
        out = self.encoder(x)
        f = self.feature(out)
        y= self.clf(out)
        return f,y

def SMILE_trainer(X, model, batch_size = 512, num_epoch=5,
                  f_temp = 0.05, p_temp = 0.15):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_con = ContrastiveLoss(batch_size = batch_size,temperature = f_temp)
    p_con = ContrastiveLoss(batch_size = model.clf_out,temperature = p_temp)
    opt = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9,weight_decay=5e-4)
    
    for k in range(num_epoch):
        model.to(device)
        n = X.shape[0]
        r = np.random.permutation(n)
        X_train = X[r,:]
        X_tensor=torch.tensor(X_train).float()
        
        losses = 0
        for j in range(n//batch_size):
            inputs = X_tensor[j*batch_size:(j+1)*batch_size,:].to(device)
            noise_inputs = inputs + torch.normal(0,1,inputs.shape).to(device)
            noise_inputs2 = inputs + torch.normal(0,1,inputs.shape).to(device)
            
            feas,o = model(noise_inputs)
            nfeas,no = model(noise_inputs2)
            
            fea_mi = f_con(feas,nfeas)
            p_mi = p_con(o.T,no.T)
            
            loss = fea_mi + p_mi
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses += loss.data.tolist()
        print("Total loss: "+str(round(losses,4)))
        gc.collect()
    
class Paired_SMILE(torch.nn.Module):
    def __init__(self,input_dim_a=2000,input_dim_b=2000,clf_out=10):
        super(Paired_SMILE, self).__init__()
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        self.clf_out = clf_out
        self.encoder_a = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim_a, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.encoder_b = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim_b, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(128, self.clf_out),
            torch.nn.Softmax(dim=1))
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(128, 32))
        
    def forward(self, x_a,x_b):
        out_a = self.encoder_a(x_a)
        f_a = self.feature(out_a)
        y_a = self.clf(out_a)
        
        out_b = self.encoder_b(x_b)
        f_b = self.feature(out_b)
        y_b = self.clf(out_b)
        return f_a,y_a,f_b,y_b
    
def PairedSMILE_trainer(X_a, X_b, model, batch_size = 512, num_epoch=5, 
                        f_temp = 0.1, p_temp = 1.0):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_con = ContrastiveLoss(batch_size = batch_size,temperature = f_temp)
    p_con = ContrastiveLoss(batch_size = model.clf_out,temperature = p_temp)
    opt = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9,weight_decay=5e-4)
    
    for k in range(num_epoch):
        
        model.to(device)
        n1 = X_a.shape[0]
        n2 = X_b.shape[0]
        print(n1,n2)
        n = min(n1, n2) #change by chunlei
        r = np.random.permutation(n)
        X_train_a = X_a[r,:]
        X_tensor_A=torch.tensor(X_train_a).float()
        X_train_b = X_b[r,:]
        X_tensor_B=torch.tensor(X_train_b).float()
        
        losses = 0
        
        for j in range(n//batch_size):
            inputs_a = X_tensor_A[j*batch_size:(j+1)*batch_size,:].to(device)
            inputs_a2 = inputs_a + torch.normal(0,1,inputs_a.shape).to(device)
            inputs_a = inputs_a + torch.normal(0,1,inputs_a.shape).to(device)
            
            inputs_b = X_tensor_B[j*batch_size:(j+1)*batch_size,:].to(device)
            inputs_b = inputs_b + torch.normal(0,1,inputs_b.shape).to(device)
            
            feas,o,nfeas,no = model(inputs_a,inputs_b)
            feas2,o2,_,_ = model(inputs_a2,inputs_b)
        
            fea_mi = f_con(feas,nfeas)+f_con(feas,feas2)
            p_mi = p_con(o.T,no.T)+p_con(o.T,o2.T)
        
            #mse_loss = mse(f_a,f_b)
            #pair = torch.ones(f_a.shape[0]).to(device)
            #cos_loss = cos(f_a,f_b,pair)
        
            loss = fea_mi + p_mi #mse_loss + 
            #loss = cos_loss * 0.5 + fea_mi + p_mi
            opt.zero_grad()
            loss.backward()
            opt.step()
        
            losses += loss.data.tolist()
        print("Total loss: "+str(round(losses,4)))
        gc.collect()
        
        
##-----------------------------------------------------------------------------
##Updates 05/09/2022
def ReferenceSMILE_trainer(X_a_paired, X_b_paired,X_a_unpaired,X_b_unpaired, model, 
                           pretrain_epoch=10,train_epoch=1000,
                           batch_size = 1024, f_temp = 0.2):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_con = ContrastiveLoss(batch_size = batch_size,temperature = f_temp)
    opt = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9,weight_decay=5e-4)
    
    for k in range(pretrain_epoch):
        model.train()
        model.to(device)
        
        n2 = X_a_unpaired.shape[0]
        r = np.random.permutation(n2)
        X_train_a_u = X_a_unpaired[r,:]
        X_tensor_Au=torch.tensor(X_train_a_u).float()
    
        n2 = X_b_unpaired.shape[0]
        r = np.random.permutation(n2)
        X_train_b_u = X_b_unpaired[r,:]
        X_tensor_Bu=torch.tensor(X_train_b_u).float()
        n= min(X_a_unpaired.shape[0],X_b_unpaired.shape[0])
        
        losses = 0
        
        for j in range(n//batch_size):
            inputs_au = X_tensor_Au[j*batch_size:(j+1)*batch_size,:].to(device)
            inputs_au2 = inputs_au + torch.normal(0,1,inputs_au.shape).to(device)
            inputs_au = inputs_au + torch.normal(0,1,inputs_au.shape).to(device)
            #inputs_au2 = torch.clone(inputs_au)
            #inputs_au2[torch.cuda.FloatTensor(inputs_au2.shape).uniform_() >= 0.2]=0 #default 0.2
            #inputs_au[torch.cuda.FloatTensor(inputs_au.shape).uniform_() >= 0.2]=0 #default 0.2
            
            inputs_bu = X_tensor_Bu[j*batch_size:(j+1)*batch_size,:].to(device)
            inputs_bu2 = inputs_bu + torch.normal(0,1,inputs_bu.shape).to(device)
            inputs_bu = inputs_bu + torch.normal(0,1,inputs_bu.shape).to(device)
            #inputs_bu2 = torch.clone(inputs_bu)
            #inputs_bu2[torch.cuda.FloatTensor(inputs_bu2.shape).uniform_() >= 0.2]=0 #default 0.2
            #inputs_bu[torch.cuda.FloatTensor(inputs_bu.shape).uniform_() >= 0.2]=0 #default 0.2
            
            feaA,feaB = model(inputs_au,inputs_bu)
            feaA2,feaB2 = model(inputs_au2,inputs_bu2)
        
            fea_mi = (f_con(feaA,feaA2)+f_con(feaB,feaB2))
            entropy = Entropy(feaA)*0.2+Entropy(feaB)*0.4
            
            loss = fea_mi + entropy
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
            losses += loss.data.tolist()
        gc.collect()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.to(device)
    
    x_a = torch.tensor(X_a_paired).float()
    x_b = torch.tensor(X_b_paired).float()
    x_a = x_a.to(device)
    x_b = x_b.to(device)
    
    opt = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9,weight_decay=5e-4)
    
    for e in range(train_epoch):
        
        A,B = model(x_a,x_b)
        
        loss = (CosineSimilarity(B,A)+CosineSimilarity(A,B)) + \
            Entropy(A)*0.2+Entropy(B)*0.4
        
        #Backward
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        gc.collect()

##-----------------------------------------------------------------------------
##Updates 06/30/2022
def Entropy(p):
    p = F.softmax(p, dim=1)
    logp = torch.log(p)
    return -(p*logp).sum(dim=1).mean()


def CosineSimilarity(p, z):
    z = z.detach() #stop gradient 
    p = F.normalize(p, dim=1) #l2-normalize 
    z = F.normalize(z, dim=1) #l2-normalize 
    return -(p*z).sum(dim=1).mean()

class littleSMILE(torch.nn.Module):
    def __init__(self,input_dim_a=30,input_dim_b=30,clf_out=10):
        super(littleSMILE, self).__init__()
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        self.clf_out = clf_out
        self.encoder_a = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim_a, self.clf_out),
            torch.nn.BatchNorm1d(self.clf_out),
            torch.nn.LeakyReLU(0.25))
        self.encoder_b = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim_b, self.clf_out),
            torch.nn.BatchNorm1d(self.clf_out),
            torch.nn.LeakyReLU(0.25))
        
    def forward(self, x_a,x_b):
        out_a = self.encoder_a(x_a)
        out_b = self.encoder_b(x_b)
        return out_a,out_b
    
def littleSMILE_trainer(x_a,x_b, model,epochs):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.to(device)
    
    x_a = x_a.to(device)
    x_b = x_b.to(device)
    
    opt = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9,weight_decay=5e-4)
    
    for e in range(epochs):
        
        A,B = model(x_a,x_b)
        
        loss = (CosineSimilarity(B,A)+CosineSimilarity(A,B)) + Entropy(A)*0.2+Entropy(B)*0.4
        
        #Backward
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        gc.collect()

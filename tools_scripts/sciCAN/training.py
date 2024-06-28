##Training network
import gc
#import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True

#import umap
import numpy as np

import random
seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

#import copy
import sys
sys.path.append('./')

from Center_loss_pytorch import CenterLoss
from contrastive_loss_pytorch import ContrastiveLoss
from network import *

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
  
def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i+=1

    return optimizer

schedule_dict = {"inv":inv_lr_scheduler}

def Cycle_train(class_num = 10, epoch=50, batch_size=512,source_trainset=None, 
                source_label=None, target_trainset=None,target_label=None):
    
    ##this training function will use label of RNA-seq data to perform semi-supervised learning
    
    cls_num_list = [np.sum(source_label == i) for i in range(class_num)]
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    
    encoder = Encoder(input_dim = source_trainset.shape[1]).cuda()
    classifier = Classifier(in_dim = 128, out_dim = class_num).cuda()
    advnet = AdvNet().cuda()
    disT = AdvNet(in_feature=target_trainset.shape[1],hidden_size=512).cuda()
    decoderS = Decoder(output_dim=target_trainset.shape[1]).cuda()
    
    config_optimizer = {"lr_type": "inv", "lr_param": {"lr": 0.001, "gamma": 0.001, "power": 0.75}}
    parameter_listC = encoder.get_parameters() + classifier.get_parameters()
    parameter_listD = encoder.get_parameters() + advnet.get_parameters() +\
        disT.get_parameters() + decoderS.get_parameters()
    optimizerC = optim.SGD(parameter_listC, lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True)
    optimizerD = optim.SGD(parameter_listD, lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True)
    
    center_loss = CenterLoss(num_classes=class_num, feat_dim=128, use_gpu=True)
    optimizer_centloss = optim.SGD([{'params': center_loss.parameters()}],lr=1e-3, 
                                   weight_decay=5e-4, momentum=0.9, nesterov=True)
    
    schedule_param = config_optimizer["lr_param"]
    lr_scheduler = schedule_dict[config_optimizer["lr_type"]]
    l1_crit = torch.nn.MSELoss()#CosineSimilarity(dim=1, eps=1e-6)#L1Loss()#
    f_con = ContrastiveLoss(batch_size = batch_size,temperature = 0.15)#0.15 default
    ###########################################################################
    for e in range(1,epoch+1):
        n = source_trainset.shape[0]
        r = np.random.permutation(n)
        X_source = torch.tensor(source_trainset[r,:]).float()
        y_source = torch.tensor(source_label[r]).long()
        
        n = target_trainset.shape[0]
        r = np.random.permutation(n)
        X_target = torch.tensor(target_trainset[r,:]).float()
        
        n=min(target_trainset.shape[0],source_trainset.shape[0])
        for j in range(n//batch_size):
            
            source_inputs = X_source[j*batch_size:(j+1)*batch_size,:].cuda()
            target_inputs = X_target[j*batch_size:(j+1)*batch_size,:].cuda()
            ntarget_inputs1 = target_inputs + torch.normal(0,1,target_inputs.shape).cuda()
            ntarget_inputs2 = target_inputs + torch.normal(0,1,target_inputs.shape).cuda()
            l = y_source[j*batch_size:(j+1)*batch_size].cuda()
            
            optimizerC = lr_scheduler(optimizerC, epoch, **schedule_param)
            optimizerD = lr_scheduler(optimizerD, epoch, **schedule_param)
            optimizerC.zero_grad()
            optimizerD.zero_grad()
            
            feature_source = encoder(source_inputs)
            feature_target1 = encoder(ntarget_inputs1)
            feature_target2 = encoder(ntarget_inputs2)
            
            output_source = classifier.forward(feature_source)
            output_target1 = classifier.forward(feature_target1)
            labels_target = torch.argmax(output_target1,1)
            #p_max, _ = torch.max(output_target1, 1)
            
            center_loss_src = center_loss(feature_target2,labels_target)*0.01
            #center_loss_src = center_loss(feature_target2[p_max>=0.5,:],labels_target[p_max>=0.5])*0.01
            classifier_loss = nn.CrossEntropyLoss(weight=per_cls_weights)(output_source, l)*2.0#use 2 default
            fea_mi = f_con(feature_target1,feature_target2)*0.25#0.25 default
            clf_loss = classifier_loss + fea_mi + center_loss_src
            clf_loss.backward()
            optimizerC.step()
            
            if center_loss_src > 0:
                for param in center_loss.parameters():
                    param.grad.data *= 0.01#(1. / centerloss_coeff)
                optimizer_centloss.step()
            
            feature_source = encoder(source_inputs)
            feature_target = encoder(target_inputs)
            gen_target = decoderS(feature_source)
            back_source = encoder(gen_target)
            
            prob_source_disT = disT.forward(target_inputs)
            prob_target_disT = disT.forward(gen_target)
            l1_loss = l1_crit(back_source,feature_source)
            
            prob_source = advnet.forward(feature_source)
            prob_target = advnet.forward(feature_target)
            prob_back = advnet.forward(back_source)
            
            wasserstein_distance = (prob_source.mean() - prob_target.mean()) * 2.0+\
                (prob_back.mean()-prob_target.mean())*3.0#use 1.5 or 1.0 for the 1st term and 2 for the 2nd default
            adv_loss = -wasserstein_distance
            wasserstein_distance_disT = prob_source_disT.mean() - prob_target_disT.mean()
            adv_loss_disT = -wasserstein_distance_disT
            
            cyc_loss = (adv_loss_disT + l1_loss)*4.0#use 2 default
            total_loss = cyc_loss + adv_loss #+ mmd_loss#
            total_loss.backward()
            optimizerD.step()
            
            ##best p: with CL 0.01,0.01; 2,0.25,2,3,4
            
        gc.collect()
                
    return encoder,classifier

def Cycle_train_wolabel(epoch=50, batch_size=512,source_trainset=None, target_trainset=None,encoder=None):
    
    ##this training function is fully unsupervised learning
    
    if encoder==None:
        encoder = scCluster(input_dim=source_trainset.shape[1],clf_out=25).cuda()
    encoder.cuda()
    advnet = AdvNet().cuda()
    disT = AdvNet(in_feature=target_trainset.shape[1],hidden_size=512).cuda()#512
    decoderS = Decoder(output_dim=target_trainset.shape[1]).cuda()
    
    config_optimizer = {"lr_type": "inv", "lr_param": {"lr": 0.001, "gamma": 0.001, "power": 0.75}}
    parameter_listC = encoder.get_parameters()
    parameter_listD = encoder.get_parameters() + advnet.get_parameters() +\
        disT.get_parameters() + decoderS.get_parameters()
    #parameter_listD = encoder.get_parameters() + disT.get_parameters() + decoderS.get_parameters()
    optimizerC = optim.SGD(parameter_listC, lr=5e-3, weight_decay=5e-4, momentum=0.9, nesterov=True)#lr=1e-3
    optimizerD = optim.SGD(parameter_listD, lr=5e-3, weight_decay=5e-4, momentum=0.9, nesterov=True)#lr=1e-3
    schedule_param = config_optimizer["lr_param"]
    lr_scheduler = schedule_dict[config_optimizer["lr_type"]]
    l1_crit = torch.nn.MSELoss()#CosineEmbeddingLoss()#L1Loss()#
    f_con = ContrastiveLoss(batch_size = batch_size,temperature = 0.15)#0.15 default
    p_con = ContrastiveLoss(batch_size = 25,temperature = 0.5)#0.5, 0.75 default
    ###########################################################################
    for e in range(1,epoch+1):
        n = source_trainset.shape[0]
        r = np.random.permutation(n)
        X_source = torch.tensor(source_trainset[r,:]).float()
        
        n = target_trainset.shape[0]
        r = np.random.permutation(n)
        X_target = torch.tensor(target_trainset[r,:]).float()
        
        n=min(target_trainset.shape[0],source_trainset.shape[0])
        for j in range(n//batch_size):
            
            source_inputs = X_source[j*batch_size:(j+1)*batch_size,:].cuda()
            nsource_inputs1 = source_inputs + torch.normal(0,1,source_inputs.shape).cuda()
            nsource_inputs2 = source_inputs + torch.normal(0,1,source_inputs.shape).cuda()
            target_inputs = X_target[j*batch_size:(j+1)*batch_size,:].cuda()
            ntarget_inputs1 = target_inputs + torch.normal(0,1,target_inputs.shape).cuda()
            ntarget_inputs2 = target_inputs + torch.normal(0,1,target_inputs.shape).cuda()
            
            optimizerC = lr_scheduler(optimizerC, epoch, **schedule_param)
            optimizerD = lr_scheduler(optimizerD, epoch, **schedule_param)
            optimizerC.zero_grad()
            optimizerD.zero_grad()
            
            _,feature_source1,output_source1 = encoder(nsource_inputs1)
            _,feature_source2,output_source2 = encoder(nsource_inputs2)
            _,feature_target1,output_target1 = encoder(ntarget_inputs1)
            _,feature_target2,output_target2 = encoder(ntarget_inputs2)
            fea_mi = f_con(feature_source1,feature_source2)*2.0+\
                f_con(feature_target1,feature_target2)*0.5##2.0 default 1st term and 0.5 default 2nd term
            p_mi = p_con(output_source1.T,output_source2.T)*2.0+\
                p_con(output_target1.T,output_target2.T)*0.5##2.0 default 1st term and 0.5 default 2nd term
            
            clf_loss = fea_mi+p_mi
            clf_loss.backward()
            optimizerC.step()
            
            feature_source,_,_ = encoder(source_inputs)
            feature_target,_,_ = encoder(target_inputs)
            gen_target = decoderS(feature_source)
            back_source,_,_ = encoder(gen_target)
            
            prob_source_disT = disT.forward(target_inputs)
            prob_target_disT = disT.forward(gen_target)
            #pair = torch.ones(back_source.shape[0]).cuda()
            l1_loss = l1_crit(back_source,feature_source)
            
            prob_source = advnet.forward(feature_source)
            prob_target = advnet.forward(feature_target)
            prob_back = advnet.forward(back_source)
            
            wasserstein_distance = (prob_source.mean() - prob_target.mean())*1.5 +\
                (prob_back.mean()-prob_target.mean())*2.5#use 1.5 or 1.0 for the 1st term and 2.5 for the 2nd default
            adv_loss = -wasserstein_distance
            wasserstein_distance_disT = prob_source_disT.mean() - prob_target_disT.mean()
            adv_loss_disT = -wasserstein_distance_disT
            
            cyc_loss = (adv_loss_disT + l1_loss)*3.0#use 3 default
            total_loss = cyc_loss + adv_loss
            total_loss.backward()
            optimizerD.step()
            
        gc.collect()
        
    return encoder

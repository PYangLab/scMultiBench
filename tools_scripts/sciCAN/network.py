#network modules

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True

import numpy as np
from easydl import aToBSheduler#*

import random
seed=1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs
    
class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply
    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)
    
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
        
class Encoder(nn.Module):
    def __init__(self,input_dim=2000):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            #nn.Dropout(0.25),
            #nn.Linear(self.input_dim, 1024),
            #nn.BatchNorm1d(1024),
            #nn.ReLU(),
            #nn.Linear(1024, 1024),
            #nn.BatchNorm1d(1024),
            #nn.ReLU(),
            nn.Linear(1000, 128),
            nn.BatchNorm1d(128),
            nn.Tanh())
            ##nn.ReLU())#,
            #nn.Dropout(0.25))
        
    def forward(self, x):
        out = self.encoder(x)
        return out
    
    def get_parameters(self):
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list
 
class scCluster(torch.nn.Module):
    def __init__(self,input_dim=2000,clf_out=10):
        super(scCluster, self).__init__()
        self.input_dim = input_dim
        self.clf_out = clf_out
        self.encoder = torch.nn.Sequential(##1000
            torch.nn.Linear(self.input_dim, 1000),
            torch.nn.BatchNorm1d(1000),
            torch.nn.ReLU(),
            #nn.Dropout(0.25),
            torch.nn.Linear(1000, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU())
            #nn.LeakyReLU(negative_slope=0.1))
            #nn.Tanh())
            #nn.Dropout(0.25))
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(128, self.clf_out),
            torch.nn.Softmax(dim=1))
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(128, 32))
        
    def forward(self, x):
        out = self.encoder(x)
        f = self.feature(out)
        y= self.clf(out)
        return out,f,y
    
    def get_parameters(self):
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list

class Decoder(nn.Module):
    def __init__(self,output_dim=2000):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.decoder = nn.Sequential(
            nn.Linear(128, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, self.output_dim))
        
    def forward(self, x):
        out = self.decoder(x)
        return out
    
    def get_parameters(self):
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list
    
class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        y = self.fc(x)
        y = self.softmax(y)
        return y

    def get_parameters(self):
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list

class AdvNet(nn.Module):
  def __init__(self, in_feature=128, hidden_size=64):
    super(AdvNet, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.norm1 = nn.BatchNorm1d(hidden_size)
    self.norm2 = nn.BatchNorm1d(hidden_size)
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0
    self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, 
                                                               gamma=10, 
                                                               max_iter=self.max_iter))

  def forward(self, x, reverse = True):
    if reverse:
        x = self.grl(x)
    x = self.ad_layer1(x)
    x = self.norm1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.norm2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1

  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]

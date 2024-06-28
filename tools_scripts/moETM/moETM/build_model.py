import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim

class encoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(encoder, self).__init__()

        self.f1 = nn.Linear(x_dim, 128)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)

        self.mu = nn.Linear(128, z_dim)
        self.log_sigma = nn.Linear(128, z_dim)

    def forward(self, x):
        h = self.dropout(self.bn1(self.act(self.f1(x))))

        mu = self.mu(h)
        log_sigma = self.log_sigma(h).clamp(-10,10)

        return mu, log_sigma

class decoder(nn.Module):
    def __init__(self, mod1_dim, mod2_dim, z_dim, emd_dim, num_batch):
        super(decoder, self).__init__()

        self.alpha_mod1 = nn.Parameter(torch.randn(mod1_dim, emd_dim))
        self.alpha_mod2 = nn.Parameter(torch.randn(mod2_dim, emd_dim))
        self.beta = nn.Parameter(torch.randn(z_dim, emd_dim))
        self.mod1_batch_bias = nn.Parameter(torch.randn(num_batch, mod1_dim))
        self.mod2_batch_bias = nn.Parameter(torch.randn(num_batch, mod2_dim))
        self.Topic_mod1 = None
        self.Topic_mod2 = None

    def forward(self, theta, batch_indices, cross_prediction = False):
        self.Topic_mod1 = torch.mm(self.alpha_mod1, self.beta.t()).t()
        self.Topic_mod2 = torch.mm(self.alpha_mod2, self.beta.t()).t()

        recon_mod1 = torch.mm(theta, self.Topic_mod1)
        recon_mod1 += self.mod1_batch_bias[batch_indices]
        if cross_prediction == False:
            recon_log_mod1 = F.log_softmax(recon_mod1, dim=-1)
        else:
            recon_log_mod1 = F.softmax(recon_mod1, dim=-1)

        recon_mod2 = torch.mm(theta, self.Topic_mod2)
        recon_mod2 += self.mod2_batch_bias[batch_indices]
        if cross_prediction == False:
            recon_log_mod2 = F.log_softmax(recon_mod2, dim=-1)
        else:
            recon_log_mod2 = F.softmax(recon_mod2, dim=-1)

        return recon_log_mod1, recon_log_mod2

class decoder_pathway(nn.Module):
    def __init__(self, mod1_dim, mod2_dim, z_dim, emd_dim, num_batch):
        super(decoder_pathway, self).__init__()

        self.alpha_mod2 = nn.Parameter(torch.randn(mod2_dim, emd_dim))
        self.beta = nn.Parameter(torch.randn(z_dim, emd_dim))
        self.mod1_batch_bias = nn.Parameter(torch.randn(num_batch, mod1_dim))
        self.mod2_batch_bias = nn.Parameter(torch.randn(num_batch, mod2_dim))
        self.Topic_mod1 = None
        self.Topic_mod2 = None

    def forward(self, theta, batch_indices, alpha_mod_gene):
        self.Topic_mod1 = torch.mm(alpha_mod_gene, self.beta.t()).t()
        self.Topic_mod2 = torch.mm(self.alpha_mod2, self.beta.t()).t()

        recon_mod1 = torch.mm(theta, self.Topic_mod1)
        recon_mod1 += self.mod1_batch_bias[batch_indices]
        recon_log_mod1 = F.log_softmax(recon_mod1,dim=-1)

        recon_mod2 = torch.mm(theta, self.Topic_mod2)
        recon_mod2 += self.mod2_batch_bias[batch_indices]
        recon_log_mod2 = F.log_softmax(recon_mod2, dim=-1)

        return recon_log_mod1, recon_log_mod2

def build_moETM(input_dim_mod1, input_dim_mod2, num_batch, num_topic=50, emd_dim=400):

    encoder_mod1 = encoder(x_dim=input_dim_mod1, z_dim=num_topic).cuda()
    encoder_mod2 = encoder(x_dim=input_dim_mod2, z_dim=num_topic).cuda()
    decoder_all = decoder(mod1_dim=input_dim_mod1, mod2_dim=input_dim_mod2, z_dim=num_topic, emd_dim=emd_dim, num_batch=num_batch).cuda()

    PARA = [{'params': encoder_mod1.parameters()},
            {'params': encoder_mod2.parameters()},
            {'params': decoder_all.parameters()}
            ]

    optimizer = optim.Adam(PARA, lr=0.001)

    return encoder_mod1, encoder_mod2, decoder_all, optimizer

def build_moETM_pathway(input_dim_mod1, input_dim_mod2, num_batch, num_topic=50, emd_dim=400):

    encoder_mod1 = encoder(x_dim=input_dim_mod1, z_dim=num_topic).cuda()
    encoder_mod2 = encoder(x_dim=input_dim_mod2, z_dim=num_topic).cuda()
    decoder_all = decoder_pathway(mod1_dim=input_dim_mod1, mod2_dim=input_dim_mod2, z_dim=num_topic, emd_dim=emd_dim, num_batch=num_batch).cuda()

    PARA = [{'params': encoder_mod1.parameters()},
            {'params': encoder_mod2.parameters()},
            {'params': decoder_all.parameters()}
            ]

    optimizer = optim.Adam(PARA, lr=0.001)

    return encoder_mod1, encoder_mod2, decoder_all, optimizer

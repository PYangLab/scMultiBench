
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time
from utils import calc_weight
from eval_utils import evaluate
import scipy
import scipy.io as sio
import pandas as pd
import os

def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


class Trainer_moETM(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
        mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu)
        return out

    def get_embed_best(self, x_mod1, x_mod2):
        self.best_encoder_mod1.eval()
        self.best_encoder_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.best_encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.best_encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu)
        return out

    def get_NLL(self, x_mod1, x_mod2, batch_indices):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

            nll_mod1 = (-recon_log_mod1 * x_mod1).sum(-1).mean()
            nll_mod2 = (-recon_log_mod2 * x_mod2).sum(-1).mean()


        return nll_mod1.item(), nll_mod2.item()

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_moETM_pathway(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer, alpha_mod_gene):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer
        self.alpha_mod_gene = alpha_mod_gene

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
        mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

        Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices, self.alpha_mod_gene)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu)
        return out

    def get_embed_best(self, x_mod1, x_mod2):
        self.best_encoder_mod1.eval()
        self.best_encoder_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.best_encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.best_encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu)
        return out

    def get_NLL(self, x_mod1, x_mod2, batch_indices):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

            nll_mod1 = (-recon_log_mod1 * x_mod1).sum(-1).mean()
            nll_mod2 = (-recon_log_mod2 * x_mod2).sum(-1).mean()


        return nll_mod1.item(), nll_mod2.item()

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

class Trainer_moETM_for_cross_prediction(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder, optimizer, direction):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder = decoder
        self.optimizer = optimizer

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None

        self.direction = direction


    def train(self, x_mod1, x_mod2, batch_indices, KL_weight):

        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder.train()

        self.optimizer.zero_grad()

        ###########################################################

        if self.direction == 'rna_to_another':
            mu_mod, log_sigma_mod = self.encoder_mod1(x_mod1)
        elif self.direction == 'another_to_rna':
            mu_mod, log_sigma_mod = self.encoder_mod2(x_mod2)
        else:
            print('Wrong direction!')

        mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod.shape[1]), use_cuda=True)


        Mu = torch.cat((mu_prior, mu_mod.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((logsigma_prior, log_sigma_mod.unsqueeze(0)), dim=0)

        mu, log_sigma = self.experts(Mu, Log_sigma)

        Theta = F.softmax(self.reparameterize(mu, log_sigma),dim=-1) #log-normal distribution

        recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

        nll_mod1 = (-recon_log_mod1*x_mod1).sum(-1).mean()
        nll_mod2 = (-recon_log_mod2*x_mod2).sum(-1).mean()

        KL = self.get_kl(mu, log_sigma).mean()
        Loss = nll_mod1 + nll_mod2 + KL_weight*KL

        Loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        return Loss.item(), nll_mod1.item(), nll_mod2.item(), KL.item()

    def reparameterize(self, mu, log_sigma):

        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    def get_embed(self, x_mod1, x_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu)
        return out

    def get_embed_best(self, x_mod1, x_mod2):
        self.best_encoder_mod1.eval()
        self.best_encoder_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.best_encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.best_encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

        out = {}
        out['delta'] = np.array(mu)
        return out

    def get_NLL(self, x_mod1, x_mod2, batch_indices):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)
            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=True)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            recon_log_mod1, recon_log_mod2 = self.decoder(Theta, batch_indices)

            nll_mod1 = (-recon_log_mod1 * x_mod1).sum(-1).mean()
            nll_mod2 = (-recon_log_mod2 * x_mod2).sum(-1).mean()


        return nll_mod1.item(), nll_mod2.item()

    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                     dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                         cast CUDA on variables
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2*logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logsigma = 0.5*torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

    def reconstruction(self, x_mod1, x_mod2, batch_indices):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(x_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(x_mod2)

            # mu_zero = torch.zeros(mu_mod1.shape)
            # log_sigma_one = torch.ones(log_sigma_mod2.shape)

            mu_prior, logsigma_prior = self.prior_expert((1, x_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution

            _, recon_mod2 = self.decoder(Theta, batch_indices, cross_prediction=True)

            Mu = torch.cat((mu_prior, mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod2.unsqueeze(0)), dim=0)
            mu, log_sigma = self.experts(Mu, Log_sigma)

            Theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)  # log-normal distribution
            recon_mod1, _ = self.decoder(Theta, batch_indices, cross_prediction=True)

            return recon_mod1, recon_mod2

def Train_moETM(trainer, Total_epoch, train_num, batch_size, Train_set, Test_set, Eval_kwargs):
    LIST = list(np.arange(0, train_num))

    X_mod1, X_mod2, batch_index = Train_set
    test_X_mod1, test_X_mod2, batch_index_test, test_adate = Test_set

    EPOCH = []
    ARI = []
    NMI = []
    ASW = []
    ASW_2 = []
    B_kBET = []
    B_ASW = []
    B_GS = []
    B_ebm = []

    best_ari = 0

    for epoch in range(Total_epoch):
        Loss_all = 0
        NLL_all_mod1 = 0
        NLL_all_mod2 = 0
        KL_all = 0

        tstart = time.time()

        np.random.shuffle(LIST)
        KL_weight = calc_weight(epoch, Total_epoch, 0, 1 / 3, 0, 1e-4)

        for iteration in range(train_num // batch_size):
            x_minibatch_mod1_T = X_mod1[LIST[iteration * batch_size: (iteration + 1) * batch_size], :].to('cuda')
            x_minibatch_mod2_T = X_mod2[LIST[iteration * batch_size: (iteration + 1) * batch_size], :].to('cuda')
            batch_minibatch_T = batch_index[LIST[iteration * batch_size: (iteration + 1) * batch_size]].to('cuda')

            loss, nll_mod1, nll_mod2, kl = trainer.train(x_minibatch_mod1_T, x_minibatch_mod2_T, batch_minibatch_T, KL_weight)

            Loss_all += loss
            NLL_all_mod1 += nll_mod1
            NLL_all_mod2 += nll_mod2
            KL_all += kl

        if (epoch % 10 == 0):

            trainer.encoder_mod1.to('cpu')
            trainer.encoder_mod2.to('cpu')

            embed = trainer.get_embed(test_X_mod1, test_X_mod2)
            test_adate.obsm.update(embed)
            Result = evaluate(adata=test_adate, n_epoch=epoch, return_fig=True, **Eval_kwargs)
            tend = time.time()
            print('epoch=%d, Time=%.4f, Cell_ARI=%.4f, Cell_NMI=%.4f, Cell_ASW=%.4f, Cell_ASW2=%.4f, Batch_KBET=%.4f, Batch_ASW=%.4f, Batch_GC=%.4f, Batch_ebm=%.4f' % (
                epoch, tend-tstart, Result['ari'], Result['nmi'], Result['asw'], Result['asw_2'], Result['k_bet'], Result['batch_asw'], Result['batch_graph_score'], Result['ebm']))

            trainer.encoder_mod1.cuda()
            trainer.encoder_mod2.cuda()

            EPOCH.append(epoch)
            ARI.append(Result['ari'])
            NMI.append(Result['nmi'])
            ASW.append(Result['asw'])
            ASW_2.append(Result['asw_2'])
            B_kBET.append(Result['k_bet'])
            B_ASW.append(Result['batch_asw'])
            B_GS.append(Result['batch_graph_score'])
            B_ebm.append(Result['ebm'])

            df = pd.DataFrame.from_dict(
                {
                    'Epoch': pd.Series(EPOCH),
                    'ARI': pd.Series(ARI),
                    'NMI': pd.Series(NMI),
                    'ASW': pd.Series(ASW),
                    'ASW_2': pd.Series(ASW_2),
                    'B_kBET': pd.Series(B_kBET),
                    'B_ASW': pd.Series(B_ASW),
                    'B_GC': pd.Series(B_GS),
                    'B_ebm': pd.Series(B_ebm),
                }
            )

            df.to_csv('./Result/moetm_all_data.csv')

            if Result['ari']>best_ari:
                best_ari = Result['ari']
                torch.save(trainer.encoder_mod1.state_dict(), './Trained_model/moetm_encoder1.pth')
                torch.save(trainer.encoder_mod2.state_dict(), './Trained_model/moetm_encoder2.pth')
                torch.save(trainer.decoder.state_dict(), './Trained_model/moetm_decoder.pth')


def Train_moETM_for_cross_prediction(trainer, Total_epoch, train_num, batch_size, Train_set, Test_set):
    LIST = list(np.arange(0, train_num))

    X_mod1, X_mod2, batch_index = Train_set
    test_X_mod1, test_X_mod2, batch_index_test, test_adate, test_mod1_sum, test_mod2_sum = Test_set

    for epoch in range(Total_epoch):
        Loss_all = 0
        NLL_all_mod1 = 0
        NLL_all_mod2 = 0
        KL_all = 0

        tstart = time.time()

        np.random.shuffle(LIST)
        KL_weight = 1e-7

        for iteration in range(train_num // batch_size):
            x_minibatch_mod1_T = X_mod1[LIST[iteration * batch_size: (iteration + 1) * batch_size], :].to('cuda')
            x_minibatch_mod2_T = X_mod2[LIST[iteration * batch_size: (iteration + 1) * batch_size], :].to('cuda')
            batch_minibatch_T = batch_index[LIST[iteration * batch_size: (iteration + 1) * batch_size]]

            loss, nll_mod1, nll_mod2, kl = trainer.train(x_minibatch_mod1_T, x_minibatch_mod2_T, batch_minibatch_T, KL_weight)

            Loss_all += loss
            NLL_all_mod1 += nll_mod1
            NLL_all_mod2 += nll_mod2
            KL_all += kl

        if epoch % 10 == 0:

            trainer.encoder_mod1.to('cpu')
            trainer.encoder_mod2.to('cpu')
            trainer.decoder.to('cpu')

            recon_mod1, recon_mod2 = trainer.reconstruction(test_X_mod1, test_X_mod2, batch_index_test)
            tend = time.time()

            if trainer.direction == 'rna_to_another':
                recon_mod = np.array(recon_mod2) * test_mod2_sum[:, np.newaxis]
                gt_mod = np.array(test_X_mod2) * test_mod2_sum[:, np.newaxis]


            elif trainer.direction == 'another_to_rna':
                recon_mod = np.array(recon_mod1) * test_mod1_sum[:, np.newaxis]
                gt_mod = np.array(test_X_mod1) * test_mod1_sum[:, np.newaxis]

            else:
                print('Wrong Direction!')

            ### save impute results
            if (epoch%100==0):
                if not os.path.exists('./recon'):  # 先检查文件夹是否存在
                    os.mkdir('./recon')
                np.save('./recon/recon_mod_epoch'+str(epoch)+'.npy',recon_mod)
                sio.savemat('./recon/recon_mod_epoch'+str(epoch)+'.mat',{'recon':recon_mod})
                np.save('./recon/gt_mod_epoch'+str(epoch)+'.npy',gt_mod)
                sio.savemat('./recon/gt_mod_epoch'+str(epoch)+'.mat',{'gt':gt_mod})


            recon_mod_tmp = np.squeeze(recon_mod.reshape([1, -1]))
            gt_mod_tmp = np.squeeze(gt_mod.reshape([1, -1]))


            recon_mod_tmp = np.log(1+recon_mod_tmp)
            gt_mod_tmp = np.log(1+gt_mod_tmp)
            Pearson = scipy.stats.pearsonr(recon_mod_tmp, gt_mod_tmp)[0]
            Spearmanr = scipy.stats.spearmanr(recon_mod_tmp, gt_mod_tmp)[0]

            print('[epoch %0d finished time %4f], Pearson_1=%.4f, Spearmanr_1=%.4f' % (epoch, tend - tstart, Pearson, Spearmanr))

            trainer.encoder_mod1.cuda()
            trainer.encoder_mod2.cuda()
            trainer.decoder.cuda()


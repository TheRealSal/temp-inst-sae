import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable

class LinearIDOL(nn.Module):
    def __init__(self,
                 x_dim,
                 z_dim, 
                 tau,
                 w,
                 noise_mode, 
                 topk_sparsity=0):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.tau = tau
        self.w = w # time-delayed contribution to Zt
        self.noise_mode = noise_mode
        self.topk_sparsity = topk_sparsity # set to 0 when l1 sparsity is used for Zt/Et (depending on noise mode)

        # Params
        self.Bs = nn.ParameterList()
        for i in range(tau):
            _B = nn.Parameter(data=torch.zeros(size=(self.z_dim, self.z_dim)),
                              requires_grad=True)
            self.Bs.append(_B)
        
        self.F_enc = nn.Parameter(data=torch.ones(size=(self.x_dim, self.z_dim)),
                                  requires_grad=True)
        self.F_dec = nn.Parameter(data=torch.ones(size=(self.z_dim, self.x_dim)),
                                  requires_grad=True)
        
        self.M = nn.Parameter(data=torch.ones(size=(self.z_dim, self.z_dim)),
                              requires_grad=True)
        
        # self.g_mus = nn.ParameterList()
        # self.g_logvars = nn.ParameterList()
        # for _ in range(tau+1):
        #     g_mu = nn.Parameter(data=torch.ones(size=(1, self.z_dim)),
        #                         requires_grad=True)
        #     g_logvar = nn.Parameter(data=torch.ones(size=(self.z_dim, self.z_dim)),
        #                             requires_grad=True)
        #     self.g_mus.append(g_mu)
        #     self.g_logvars.append(g_logvar)
        self.init_params()
    
    def init_params(self):
        nn.init.xavier_normal_(self.F_enc.data)
        nn.init.xavier_normal_(self.F_dec.data)
        nn.init.xavier_normal_(self.M.data)
        # for B in self.Bs:
        #     nn.init.xavier_normal_(B.data)
        # for net in self.G_lin_mus:
        #     nn.init.xavier_normal_(net.data)
        # for net in self.G_lin_logvars:
        #     nn.init.xavier_normal_(net.data)

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps
    
    def forward(self, Xp, enable_w=False):
        '''
        Xp: torch.tensor -- batch_size x z_dim(768) x (tau + 1); p means the time period of tau + 1 
        '''
        batch_size, z_dim, p = Xp.shape
        # assert z_dim == self.z_dim

        # disable topk sparsity during evaluation
        if not self.training:
            self.topk_sparsity = 0

        # Recons Xp by MSE -> constrain on A(==F_enc^{-1}) to be invertible
        Zp = torch.einsum('hd,bdt->bht', self.F_enc.T, Xp)
        recons_Xp = torch.einsum('dh,bht->bdt', self.F_dec.T, Zp)
        loss_mse_Xt = F.mse_loss(input=recons_Xp[:,:,-1], target=Xp[:,:,-1])

        # Recons Zt by MSE; note that here Zt is the last time step of p, instead of Zp
        loss_sparse_Bs = 0.
        M = torch.tril(self.M, diagonal=1) # the instantaneous relations
        if enable_w:
            w = self.w
            _w = 1. - self.w
        else:
            w = 1.
            _w = 1.
        Zt = _w * torch.einsum('hd,bd->bh', M, Zp[:, :, self.tau])  # instantaneous effect
        for lag in range(1, self.tau+1):
            B_lag = self.Bs[lag-1]
            loss_sparse_Bs = loss_sparse_Bs + F.l1_loss(B_lag, torch.zeros_like(B_lag))
            Zt_lag = Zp[:, :, self.tau-lag] 
            Zt = Zt + w * torch.einsum('hd,bd->bh', B_lag, Zt_lag) # time-delayed effect
        
        # Use sparse Zt going forward
        if self.topk_sparsity > 0:
            Zt_abs = torch.abs(Zt)
            topk_vals, topk_indices = torch.topk(Zt_abs, self.topk_sparsity, dim=1)
            mask = torch.zeros_like(Zt)
            mask.scatter_(1, topk_indices, 1.0)
            Zt = Zt * mask  
        
        loss_mse_Zt = F.mse_loss(input=Zt, target=Zp[:, :, self.tau])
            
        # Independent Et; Et = Zt - \sum B_\tau Z_{t-\tau} - M Zt 
        Et = Zp[:, :, self.tau] - Zt
        if self.noise_mode == 'gau':
            logp_Et_normal = -torch.trace(torch.cov(Et))
            loss_indep = -logp_Et_normal
        elif self.noise_mode == 'lap':
            logp_Et_lap = -F.l1_loss(Et, torch.zeros_like(Et))
            loss_indep = -logp_Et_lap
        else: 
            raise NotImplementedError

        # Sparsity on B (calculated before), M, and Z
        # Sparsity on M
        loss_sparse_M = F.l1_loss(M, torch.zeros_like(M))
        
        # Sparsity on Z
        loss_sparse_Zt = F.l1_loss(Zt, torch.zeros_like(Zt))

        # Return results
        return loss_mse_Xt, loss_mse_Zt, loss_indep, loss_sparse_Bs, loss_sparse_M, loss_sparse_Zt
        
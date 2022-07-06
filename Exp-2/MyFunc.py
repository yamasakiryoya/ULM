#imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from numpy import mean,std,exp,sqrt
from scipy.special import comb


#====================  tool  ====================#
def K_mean(arr, K):
    n = len(arr)
    tmp = np.zeros(n)
    for i in range(n):
        l = np.max([0,i-K])
        r = np.min([n,i+K+1])
        tmp[i] = np.mean(np.nan_to_num(arr[l:r], nan=10.**6))
    return tmp
def back_argmin(arr):
    #argmin
    return len(arr)-1-np.argmin(np.nan_to_num(arr[::-1], nan=10.**6))


#====================  labeling  ====================#
#labeling function
def labeling_Z(pro, device, K):
    #task-Z
    return pro.argmax(dim=1)
def labeling_A(pro, device, K):
    #task-A
    tmp = torch.zeros(K,K).to(device)
    for j in range(K):
        for k in range(K):
            tmp[j,k] = np.fabs(j-k)
    res = torch.argmin(torch.mm(pro, tmp), dim=1)
    return torch.LongTensor(res).to(device)
def labeling_S(pro, device, K):
    #task-S
    tmp = torch.zeros(K,K).to(device)
    for j in range(K):
        for k in range(K):
            tmp[j,k] = (j-k)**2
    res = torch.argmin(torch.mm(pro, tmp), dim=1)
    return torch.LongTensor(res).to(device)


#====================  models  ====================#
#proportional odds models#
class PO_ORD_CL(nn.Module):
    def __init__(self, d, M, K):
        super(PO_ORD_CL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.tmp_bi  = nn.Parameter(torch.zeros(K-2).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        bi = torch.zeros(self.K-1).float()
        for k in range(1, self.K-1):
            bi[k] = bi[k-1] + torch.exp(self.tmp_bi[k-1])
        #
        tmp = bi - g
        tmp = torch.cat([-10.**6*torch.ones(tmp.shape[0],1).float(), tmp, 10.**6*torch.ones(tmp.shape[0],1).float()], dim=1)
        pro = torch.sigmoid(tmp[:,1:]) - torch.sigmoid(tmp[:,:-1])
        return pro
class PO_ACL(nn.Module):
    def __init__(self, d, M, K):
        super(PO_ACL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.tmp_bi  = nn.Parameter(torch.arange(1,K-1).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        bi = torch.cat([torch.tensor([0.]), self.tmp_bi])
        #
        tmp1 = bi - g
        tmp2 = torch.zeros(tmp1.shape[0], self.K).float()
        for k in range(1,self.K):
            tmp2[:,k] = tmp2[:,k-1] + tmp1[:,k-1]
        pro = -tmp2
        pro = pro.softmax(dim=1)
        return pro
class PO_CRL(nn.Module):
    def __init__(self, d, M, K):
        super(PO_CRL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.tmp_bi  = nn.Parameter(torch.arange(1,K-1).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        bi = torch.cat([torch.tensor([0.]), self.tmp_bi])
        #
        tmp = bi - g
        tmp = torch.cat([tmp, 10.**6*torch.ones(tmp.shape[0],1).float()], dim=1)
        pro = torch.ones(tmp.shape[0], self.K).float()
        for k in range(self.K):
            for j in range(k):
                pro[:,k] *= torch.sigmoid(-tmp[:,j])
            pro[:,k] *= torch.sigmoid(tmp[:,k])
        return pro
#heteroscedastic models#
class OH_ORD_CL(nn.Module):
    def __init__(self, d, M, K):
        super(OH_ORD_CL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.s1 = nn.Linear(d, M); torch.nn.init.normal_(self.s1.weight, mean=0., std=.1); torch.nn.init.normal_(self.s1.bias, mean=0., std=.1)
        self.s2 = nn.Linear(M, M); torch.nn.init.normal_(self.s2.weight, mean=0., std=.1); torch.nn.init.normal_(self.s2.bias, mean=0., std=.1)
        self.s3 = nn.Linear(M, M); torch.nn.init.normal_(self.s3.weight, mean=0., std=.1); torch.nn.init.normal_(self.s3.bias, mean=0., std=.1)
        self.s4 = nn.Linear(M, 1); torch.nn.init.normal_(self.s4.weight, mean=0., std=.1); torch.nn.init.normal_(self.s4.bias, mean=0., std=.1)
        self.tmp_bi  = nn.Parameter(torch.zeros(K-2).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        s = torch.sigmoid(self.s1(x))
        s = torch.sigmoid(self.s2(s))
        s = torch.sigmoid(self.s3(s))
        s = 1.0+torch.exp(self.s4(s)/10.)
        #
        bi = torch.zeros(self.K-1).float()
        for k in range(1, self.K-1):
            bi[k] = bi[k-1] + torch.exp(self.tmp_bi[k-1])
        #
        tmp = (bi - g)/s
        tmp = torch.cat([-10.**6*torch.ones(tmp.shape[0],1).float(), tmp, 10.**6*torch.ones(tmp.shape[0],1).float()], dim=1)
        pro = torch.sigmoid(tmp[:,1:]) - torch.sigmoid(tmp[:,:-1])
        return pro
class OH_ACL(nn.Module):
    def __init__(self, d, M, K):
        super(OH_ACL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.s1 = nn.Linear(d, M); torch.nn.init.normal_(self.s1.weight, mean=0., std=.1); torch.nn.init.normal_(self.s1.bias, mean=0., std=.1)
        self.s2 = nn.Linear(M, M); torch.nn.init.normal_(self.s2.weight, mean=0., std=.1); torch.nn.init.normal_(self.s2.bias, mean=0., std=.1)
        self.s3 = nn.Linear(M, M); torch.nn.init.normal_(self.s3.weight, mean=0., std=.1); torch.nn.init.normal_(self.s3.bias, mean=0., std=.1)
        self.s4 = nn.Linear(M, 1); torch.nn.init.normal_(self.s4.weight, mean=0., std=.1); torch.nn.init.normal_(self.s4.bias, mean=0., std=.1)
        self.tmp_bi  = nn.Parameter(torch.arange(1,K-1).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        s = torch.sigmoid(self.s1(x))
        s = torch.sigmoid(self.s2(s))
        s = torch.sigmoid(self.s3(s))
        s = 1.0+torch.exp(self.s4(s)/10.)
        #
        bi = torch.cat([torch.tensor([0.]), self.tmp_bi])
        #
        tmp1 = (bi - g)/s
        tmp2 = torch.zeros(tmp1.shape[0], self.K).float()
        for k in range(1,self.K):
            tmp2[:,k] = tmp2[:,k-1] + tmp1[:,k-1]
        pro = -tmp2
        pro = pro.softmax(dim=1)
        return pro
class OH_CRL(nn.Module):
    def __init__(self, d, M, K):
        super(OH_CRL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.s1 = nn.Linear(d, M); torch.nn.init.normal_(self.s1.weight, mean=0., std=.1); torch.nn.init.normal_(self.s1.bias, mean=0., std=.1)
        self.s2 = nn.Linear(M, M); torch.nn.init.normal_(self.s2.weight, mean=0., std=.1); torch.nn.init.normal_(self.s2.bias, mean=0., std=.1)
        self.s3 = nn.Linear(M, M); torch.nn.init.normal_(self.s3.weight, mean=0., std=.1); torch.nn.init.normal_(self.s3.bias, mean=0., std=.1)
        self.s4 = nn.Linear(M, 1); torch.nn.init.normal_(self.s4.weight, mean=0., std=.1); torch.nn.init.normal_(self.s4.bias, mean=0., std=.1)
        self.tmp_bi  = nn.Parameter(torch.arange(1,K-1).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        s = torch.sigmoid(self.s1(x))
        s = torch.sigmoid(self.s2(s))
        s = torch.sigmoid(self.s3(s))
        s = 1.0+torch.exp(self.s4(s)/10.)
        #
        bi = torch.cat([torch.tensor([0.]), self.tmp_bi])
        #
        tmp = (bi - g)/s
        tmp = torch.cat([tmp, 10.**6*torch.ones(tmp.shape[0],1).float()], dim=1)
        pro = torch.ones(tmp.shape[0], self.K).float()
        for k in range(self.K):
            for j in range(k):
                pro[:,k] *= torch.sigmoid(-tmp[:,j])
            pro[:,k] *= torch.sigmoid(tmp[:,k])
        return pro
#distribution free models#
class ORD_CL(nn.Module):
    def __init__(self, d, M, K):
        super(ORD_CL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, K-1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        tmp = self.g4(g)
        #
        g = torch.zeros(tmp.shape[0], self.K-1).float()
        g[:,0] = tmp[:,0]
        for k in range(1, self.K-1):
            g[:,k] = g[:,k-1] + torch.exp(tmp[:,k])
        g = torch.cat([-10.**6*torch.ones(g.shape[0],1).float(), g, 10.**6*torch.ones(g.shape[0],1).float()], dim=1)
        pro = torch.sigmoid(g[:,1:]) - torch.sigmoid(g[:,:-1])
        return pro
class ACL(nn.Module):
    def __init__(self, d, M, K):
        super(ACL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, K-1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        tmp = torch.zeros(g.shape[0], self.K).float()
        for k in range(1,self.K):
            tmp[:,k] = tmp[:,k-1] + g[:,k-1]
        pro = -tmp
        pro = pro.softmax(dim=1)
        return pro
class CRL(nn.Module):
    def __init__(self, d, M, K):
        super(CRL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, K-1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        g = torch.cat([g, 10.**6*torch.ones(g.shape[0],1).float()], dim=1)
        pro = torch.ones(g.shape[0], self.K).float()
        for k in range(self.K):
            for j in range(k):
                pro[:,k] *= torch.sigmoid(-g[:,j])
            pro[:,k] *= torch.sigmoid(g[:,k])
        return pro
class SL(nn.Module):
    def __init__(self, d, M, K):
        super(SL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        #self.g4 = nn.Linear(M, K-1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, K); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        #pro = torch.cat([torch.zeros(g.shape[0],1).float(),g], dim=1)
        #pro = pro.softmax(dim=1)
        pro = g.softmax(dim=1)
        return pro
#ordered acl models#
class PO_ORD_ACL(nn.Module):
    def __init__(self, d, M, K):
        super(PO_ORD_ACL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.tmp_bi  = nn.Parameter(torch.zeros(K-2).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        bi = torch.zeros(self.K-1).float()
        for k in range(1, self.K-1):
            bi[k] = bi[k-1] + torch.exp(self.tmp_bi[k-1])
        #
        tmp1 = bi - g
        tmp2 = torch.zeros(tmp1.shape[0], self.K).float()
        for k in range(1,self.K):
            tmp2[:,k] = tmp2[:,k-1] + tmp1[:,k-1]
        pro = -tmp2
        pro = pro.softmax(dim=1)
        return pro
class OH_ORD_ACL(nn.Module):
    def __init__(self, d, M, K):
        super(OH_ORD_ACL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.s1 = nn.Linear(d, M); torch.nn.init.normal_(self.s1.weight, mean=0., std=.1); torch.nn.init.normal_(self.s1.bias, mean=0., std=.1)
        self.s2 = nn.Linear(M, M); torch.nn.init.normal_(self.s2.weight, mean=0., std=.1); torch.nn.init.normal_(self.s2.bias, mean=0., std=.1)
        self.s3 = nn.Linear(M, M); torch.nn.init.normal_(self.s3.weight, mean=0., std=.1); torch.nn.init.normal_(self.s3.bias, mean=0., std=.1)
        self.s4 = nn.Linear(M, 1); torch.nn.init.normal_(self.s4.weight, mean=0., std=.1); torch.nn.init.normal_(self.s4.bias, mean=0., std=.1)
        self.tmp_bi  = nn.Parameter(torch.zeros(K-2).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        s = torch.sigmoid(self.s1(x))
        s = torch.sigmoid(self.s2(s))
        s = torch.sigmoid(self.s3(s))
        s = 1.0+torch.exp(self.s4(s)/10.)
        #
        bi = torch.zeros(self.K-1).float()
        for k in range(1, self.K-1):
            bi[k] = bi[k-1] + torch.exp(self.tmp_bi[k-1])
        #
        tmp1 = (bi - g)/s
        tmp2 = torch.zeros(tmp1.shape[0], self.K).float()
        for k in range(1,self.K):
            tmp2[:,k] = tmp2[:,k-1] + tmp1[:,k-1]
        pro = -tmp2
        pro = pro.softmax(dim=1)
        return pro
class ORD_ACL(nn.Module):
    def __init__(self, d, M, K):
        super(ORD_ACL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, K-1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        tmp1 = self.g4(g)
        #
        g = torch.zeros(tmp1.shape[0], self.K-1).float()
        g[:,0] = tmp1[:,0]
        for k in range(1, self.K-1):
            g[:,k] = g[:,k-1] + torch.exp(tmp1[:,k])
        #
        tmp2 = torch.zeros(g.shape[0], self.K).float()
        for k in range(1,self.K):
            tmp2[:,k] = tmp2[:,k-1] + g[:,k-1]
        pro = -tmp2
        pro = pro.softmax(dim=1)
        return pro
#squared ordered sl models#
class PO_VS_SL(nn.Module):
    def __init__(self, d, M, K):
        super(PO_VS_SL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.tmp_bi  = nn.Parameter(torch.zeros(K-1).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        bi = torch.zeros(self.K).float()
        for k in range(1, self.K):
            bi[k] = bi[k-1] + torch.exp(self.tmp_bi[k-1])
        #
        tmp = bi - g
        pro = -torch.square(tmp)
        pro = pro.softmax(dim=1)
        return pro
class OH_VS_SL(nn.Module):
    def __init__(self, d, M, K):
        super(OH_VS_SL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.s1 = nn.Linear(d, M); torch.nn.init.normal_(self.s1.weight, mean=0., std=.1); torch.nn.init.normal_(self.s1.bias, mean=0., std=.1)
        self.s2 = nn.Linear(M, M); torch.nn.init.normal_(self.s2.weight, mean=0., std=.1); torch.nn.init.normal_(self.s2.bias, mean=0., std=.1)
        self.s3 = nn.Linear(M, M); torch.nn.init.normal_(self.s3.weight, mean=0., std=.1); torch.nn.init.normal_(self.s3.bias, mean=0., std=.1)
        self.s4 = nn.Linear(M, 1); torch.nn.init.normal_(self.s4.weight, mean=0., std=.1); torch.nn.init.normal_(self.s4.bias, mean=0., std=.1)
        self.tmp_bi  = nn.Parameter(torch.zeros(K-1).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        s = torch.sigmoid(self.s1(x))
        s = torch.sigmoid(self.s2(s))
        s = torch.sigmoid(self.s3(s))
        s = 1.0+torch.exp(self.s4(s)/10.)
        #
        bi = torch.zeros(self.K).float()
        for k in range(1, self.K):
            bi[k] = bi[k-1] + torch.exp(self.tmp_bi[k-1])
        #
        tmp = (bi - g)/s
        pro = -torch.square(tmp)
        pro = pro.softmax(dim=1)
        return pro
class VS_SL(nn.Module):
    def __init__(self, d, M, K):
        super(VS_SL, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, K); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        tmp = torch.zeros(g.shape[0], self.K).float()
        tmp[:,0] = g[:,0]
        for k in range(1, self.K):
            tmp[:,k] = tmp[:,k-1] + torch.exp(g[:,k])
        #
        pro = -torch.square(tmp)
        pro = pro.softmax(dim=1)
        return pro
#bin model
class BIN(nn.Module):
    def __init__(self, d, M, K):
        super(BIN, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.s  = nn.Parameter(torch.zeros(1).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = torch.sigmoid(self.g4(g))
        #
        pro = torch.zeros(g.shape[0], self.K).float()
        for k in range(self.K):
            pro[:,k] = comb(int(self.K-1), k, exact=True)*torch.pow(g[:,0], k)*torch.pow(1.-g[:,0], int(self.K-1-k))
        pro = torch.log(pro)/torch.exp(self.s)
        pro = pro.softmax(dim=1)
        return pro
class OH_BIN(nn.Module):
    def __init__(self, d, M, K):
        super(OH_BIN, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.s1 = nn.Linear(d, M); torch.nn.init.normal_(self.s1.weight, mean=0., std=.1); torch.nn.init.normal_(self.s1.bias, mean=0., std=.1)
        self.s2 = nn.Linear(M, M); torch.nn.init.normal_(self.s2.weight, mean=0., std=.1); torch.nn.init.normal_(self.s2.bias, mean=0., std=.1)
        self.s3 = nn.Linear(M, M); torch.nn.init.normal_(self.s3.weight, mean=0., std=.1); torch.nn.init.normal_(self.s3.bias, mean=0., std=.1)
        self.s4 = nn.Linear(M, 1); torch.nn.init.normal_(self.s4.weight, mean=0., std=.1); torch.nn.init.normal_(self.s4.bias, mean=0., std=.1)
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = torch.sigmoid(self.g4(g))
        #
        s = torch.sigmoid(self.s1(x))
        s = torch.sigmoid(self.s2(s))
        s = torch.sigmoid(self.s3(s))
        s = 0.01+torch.exp(self.s4(s)/10.)
        #
        pro = torch.zeros(g.shape[0], self.K).float()
        for k in range(self.K):
            pro[:,k] = comb(int(self.K-1), k, exact=True)*torch.pow(g[:,0], k)*torch.pow(1.-g[:,0], int(self.K-1-k))
        pro = torch.log(pro)/s
        pro = pro.softmax(dim=1)
        return pro
class POI(nn.Module):
    def __init__(self, d, M, K):
        super(POI, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.s  = nn.Parameter(torch.zeros(1).float())
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        pro = torch.zeros(g.shape[0], self.K).float()
        tmp = 1.
        for k in range(self.K):
            if k!=0: tmp += np.log(k)
            pro[:,k] = tmp-k*g[:,0]
        pro = -pro/torch.exp(self.s)
        pro = pro.softmax(dim=1)
        return pro

class OH_POI(nn.Module):
    def __init__(self, d, M, K):
        super(OH_POI, self).__init__()
        self.g1 = nn.Linear(d, M); torch.nn.init.normal_(self.g1.weight, mean=0., std=.1); torch.nn.init.normal_(self.g1.bias, mean=0., std=.1)
        self.g2 = nn.Linear(M, M); torch.nn.init.normal_(self.g2.weight, mean=0., std=.1); torch.nn.init.normal_(self.g2.bias, mean=0., std=.1)
        self.g3 = nn.Linear(M, M); torch.nn.init.normal_(self.g3.weight, mean=0., std=.1); torch.nn.init.normal_(self.g3.bias, mean=0., std=.1)
        self.g4 = nn.Linear(M, 1); torch.nn.init.normal_(self.g4.weight, mean=0., std=.1); torch.nn.init.normal_(self.g4.bias, mean=0., std=.1)
        self.s1 = nn.Linear(d, M); torch.nn.init.normal_(self.s1.weight, mean=0., std=.1); torch.nn.init.normal_(self.s1.bias, mean=0., std=.1)
        self.s2 = nn.Linear(M, M); torch.nn.init.normal_(self.s2.weight, mean=0., std=.1); torch.nn.init.normal_(self.s2.bias, mean=0., std=.1)
        self.s3 = nn.Linear(M, M); torch.nn.init.normal_(self.s3.weight, mean=0., std=.1); torch.nn.init.normal_(self.s3.bias, mean=0., std=.1)
        self.s4 = nn.Linear(M, 1); torch.nn.init.normal_(self.s4.weight, mean=0., std=.1); torch.nn.init.normal_(self.s4.bias, mean=0., std=.1)
        self.K = K
    def forward(self, x):
        g = torch.sigmoid(self.g1(x))
        g = torch.sigmoid(self.g2(g))
        g = torch.sigmoid(self.g3(g))
        g = self.g4(g)
        #
        s = torch.sigmoid(self.s1(x))
        s = torch.sigmoid(self.s2(s))
        s = torch.sigmoid(self.s3(s))
        s = 0.01+torch.exp(self.s4(s)/10.)
        #
        pro = torch.zeros(g.shape[0], self.K).float()
        tmp = 1.
        for k in range(self.K):
            if k!=0: tmp += np.log(k)
            pro[:,k] = tmp-k*g[:,0]
        pro = -pro/s
        pro = pro.softmax(dim=1)
        return pro


#====================  models  ====================#
#training function#
def train_func(model, device, loader, optimizer):
    model.train()
    for batch_idx, (X, Y) in enumerate(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pro = model(X)
        pro = torch.clamp(pro, min=.1**30, max=1.-.1**30)
        loss = F.nll_loss(torch.log(pro), Y)
        loss.backward()
        optimizer.step()
#validation/test function#
def test_func(model, device, loader, K, CV=None):
    model.eval()
    n, loss = 0, 0.
    MZE_Z, MAE_Z, MSE_Z, MZE_A, MAE_A, MSE_A, MZE_S, MAE_S, MSE_S = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            pro = model(X)
            pro = torch.clamp(pro, min=.1**30, max=1.-.1**30)
            loss += F.nll_loss(torch.log(pro), Y, reduction='sum')
            #
            pre_Z = labeling_Z(pro, device, K)
            MZE_Z += torch.sum(Y != pre_Z).float()
            if CV!='CV': MAE_Z += torch.sum(torch.abs(Y.float() - pre_Z.float())).float()
            if CV!='CV': MSE_Z += torch.sum(torch.square(Y.float() - pre_Z.float())).float()
            #
            pre_A = labeling_A(pro, device, K)
            if CV!='CV': MZE_A += torch.sum(Y != pre_A).float()
            MAE_A += torch.sum(torch.abs(Y.float() - pre_A.float())).float()
            if CV!='CV': MSE_A += torch.sum(torch.square(Y.float() - pre_A.float())).float()
            #
            pre_S = labeling_S(pro, device, K)
            if CV!='CV': MZE_S += torch.sum(Y != pre_S).float()
            if CV!='CV': MAE_S += torch.sum(torch.abs(Y.float() - pre_S.float())).float()
            MSE_S += torch.sum(torch.square(Y.float() - pre_S.float())).float()
            #
            n += len(Y)
    loss = loss/n
    MZE_Z, MAE_Z, MSE_Z = MZE_Z/n, MAE_Z/n, MSE_Z/n
    MZE_A, MAE_A, MSE_A = MZE_A/n, MAE_A/n, MSE_A/n
    MZE_S, MAE_S, MSE_S = MZE_S/n, MAE_S/n, MSE_S/n
    #out: 10
    return loss, MZE_Z, MAE_Z, sqrt(MSE_Z), MZE_A, MAE_A, sqrt(MSE_A), MZE_S, MAE_S, sqrt(MSE_S)


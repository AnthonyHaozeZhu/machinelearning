from typing import Optional

import pandas as pd
from torch.nn import Linear, BatchNorm1d
import torch
import torch.nn as nn
from torch import Tensor, lgamma
HASGPU = torch.has_cuda

class LZINBLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X: Tensor, PI: Tensor = None, M: Tensor = None, THETA: Tensor = None):
        # 防止出现除0，log(0) log (负数) 等等等
        eps = self.eps
        # deal with inf
        max1 = max(THETA.max(), M.max())
        if THETA.isinf().sum() != 0:
            THETA = torch.where(THETA.isinf(), torch.full_like(THETA, max1), THETA)
        if M.isinf().sum() != 0:
            M = torch.where(M.isinf(), torch.full_like(M, max1), M)


        if PI.isnan().sum() != 0:
            PI = torch.where(PI.isnan(), torch.full_like(PI, eps), PI)
        if THETA.isnan().sum() != 0:
            THETA = torch.where(THETA.isnan(), torch.full_like(THETA, eps), THETA)
        if M.isnan().sum() != 0:
            M = torch.where(M.isnan(), torch.full_like(M, eps), M)

        # 之前的
        # lnb = lgamma(X + THETA + eps) - lgamma(THETA + eps) + \
        #       THETA * (torch.log(THETA + eps) - torch.log(M + THETA + eps)) + \
        #       X * (torch.log(M + eps) - torch.log(M + THETA + eps))
        # lnb = torch.where(lnb > 0, torch.full_like(lnb, 0.), lnb)
        # assert (lnb > 0).sum() == 0, 'ln(nb) greater than 0'
        # nb = torch.exp(lnb)
        # zinb_nonZeroCase = (1 - PI) * nb + eps
        # zinb_zeroCase = PI + (1 - PI) * nb + eps
        # zinb = torch.where(torch.less(X, 1e-10), zinb_zeroCase, zinb_nonZeroCase)
        # # python 有链式比较！！！
        # # 等价于 0 < zinb and zinb <= 1(概率值)
        # assert (zinb <= 0).sum() >= 0, 'zinb less than 0'
        # assert (zinb > 1.1).sum() >= 0, 'zinb bigger than 1.1'
        # # 最大化概率密度 即 最小化负概率密度 即 最小化loss
        # loss = torch.log(zinb).sum()
        # loss = -loss
        # return loss

        # #final版本
        eps = torch.tensor(1e-10)
        # 事实上u即为y_pred，即补差的值
        # 注意是负对数，因此我们可以将乘法和除法变为加减法，
        THETA = torch.minimum(THETA, torch.tensor(1e6))
        t1 = torch.lgamma(THETA + eps) + torch.lgamma(X + 1.0) - torch.lgamma(X + THETA + eps)
        t2 = (THETA + X) * torch.log(1.0 + (M / (THETA + eps))) + (X * (torch.log(THETA + eps) - torch.log(M + eps)))
        nb = t1 + t2
        nb = torch.where(torch.isnan(nb), torch.zeros_like(nb) + max1, nb)
        nb_case = nb - torch.log(1.0 - PI + eps)
        zero_nb = torch.pow(THETA / (THETA + M + eps), THETA)
        zero_case = -torch.log(PI + ((1.0 - PI) * zero_nb) + eps)
        res = torch.where(torch.less(X, 1e-8), zero_case, nb_case)
        res = torch.where(torch.isnan(res), torch.zeros_like(res) + max1, res)
        return torch.mean(res)



class AutoEncoder(nn.Module):
    def __init__(self, input_size=None, hasBN=False):
        """
        该autoencoder还可以改进：
        dropout层

        :param input_size:
        """
        super().__init__()
        self.intput_size = input_size
        self.hasBN=hasBN
        self.input = Linear(input_size, 64)
        self.bn1 = BatchNorm1d(64) if hasBN else None
        self.encode = Linear(64, 32)
        self.bn2 = BatchNorm1d(32) if hasBN else None
        self.decode = Linear(32, 64)
        self.bn3 = BatchNorm1d(64) if hasBN else None
        self.out_put_PI = Linear(64, input_size)
        self.out_put_M = Linear(64, input_size)
        self.out_put_THETA = Linear(64, input_size)
        self.reLu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = preprocess_data(x)
        x = self.input(x)
        if self.hasBN:x = self.bn1(x)
        x = self.reLu(x)
        x = self.encode(x)
        if self.hasBN:x = self.bn2(x)
        x = self.reLu(x)
        x = self.decode(x)
        if self.hasBN:x = self.bn3(x)
        x = self.reLu(x)
        PI = self.sigmoid(self.out_put_PI(x))
        M = torch.exp(self.out_put_M(x))
        THETA = torch.exp(self.out_put_THETA(x))
        return PI, M, THETA


def preprocess_data(data: Tensor):
    """
    预处理数据 按照X-bar公式 zscore处理
    :param data:
    :return: 处理得到的数据
    """
    s = torch.sum(data,dim=1)
    median = torch.median(s)
    s = s/median
    s = torch.inverse(torch.diag(s))
    norm_data = torch.matmul(s, data) + 1
    norm_data = torch.log(norm_data)
    norm_data = (norm_data - norm_data.mean()) / norm_data.std()
    return norm_data

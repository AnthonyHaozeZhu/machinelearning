from math import floor

import pandas as pd
import torch
import math
from torch.nn.utils import clip_grad_norm_

from Autoencoder import AutoEncoder, LZINBLoss, HASGPU
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# 超参数
BATCHSIZE = 32
EPS = 1e-9

BETA1=0.99
BETA2=0.999
WEIGHT_DECAY = 1e-6
EPOCH_NUM = 3000
EVER_SAVING = 1000

LR = 5e-4
START_EPOCH = 0

FILE_NAME = 'data.csv'

f = open('diary.txt','w')
def train(EPOCH_NUM=100, print_batchloss=False, autoencoder=None, loader=None, startEpoch=0,ever_saving = EVER_SAVING):
    """

    :param print_batchloss: 是否打印batch训练信息，默认为 False 否
    """
    lzinbloss = LZINBLoss(eps=EPS)
    opt = Adam(autoencoder.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WEIGHT_DECAY)
    # opt = SGD(autoencoder.parameters(), lr=1e-2, momentum=0.8)
    mean_loss=0
    for epoch in range(EPOCH_NUM+1):
        epoch_loss = 0

        for batch, batch_data in enumerate(loader):
            # 一个batch拿32条数据
            opt.zero_grad()
            # d是原始计数矩阵
            d = batch_data[0]
            # 论文内容 照着实现
            # 正向传播计算损失函数当前值
            PI, M, THETA = autoencoder(d)
            templ = lzinbloss(d, PI, M, THETA)
            epoch_loss += templ
            if print_batchloss:
                print(f'epoch:{epoch+startEpoch},batch:{batch},batch loss:{templ},(batch size {BATCHSIZE})')
                f.write(f'epoch:{epoch+startEpoch},batch:{batch},batch loss:{templ},(batch size {BATCHSIZE})\n')
            # 反向传播计算梯度
            templ.backward()
            # 截取最大梯度
            clip_grad_norm_(autoencoder.parameters(), max_norm=5, norm_type=2)
            # 梯度下降
            opt.step()

        print(f'epoch:{epoch+startEpoch},epoch loss:{epoch_loss}')
        f.write(f'epoch:{epoch+startEpoch},epoch loss:{mean_loss}\n')
        # mean_loss=0
        if epoch % ever_saving == 0 and epoch!=0: torch.save(autoencoder.state_dict(), open(f'0114epoch{epoch+startEpoch}.pkl', 'wb'))
        # if epoch % ever_saving == 0 and epoch!= 0 : torch.save(autoencoder.state_dict(), open(f'0113epoch{epoch+startEpoch}withoutBN.pkl', 'wb'))


if __name__ == '__main__':
    autoencoder = AutoEncoder(1000,hasBN=True)
    start_epoch = START_EPOCH
    if start_epoch != 0 : autoencoder.load_state_dict(torch.load(f'0113epoch{start_epoch}.pkl'))
    # if start_epoch != 0 : autoencoder.load_state_dict(torch.load(f'0113epoch{start_epoch}withoutBN.pkl'))
    data = pd.read_csv(FILE_NAME).astype('float32')
    data = torch.from_numpy(data.values.T)
    len=math.floor(0.7*(len(data)))
    data=data[len:]

    # if HASGPU:
    #     autoencoder = autoencoder.cuda()
    #     lzinbloss = lzinbloss.cuda()
    #     data = data.cuda()

    # 稍微修改
    data = TensorDataset(data)
    loader = DataLoader(data, BATCHSIZE, True)

    train(EPOCH_NUM=EPOCH_NUM,print_batchloss=False,autoencoder=autoencoder,loader=loader,startEpoch=start_epoch,ever_saving=EVER_SAVING)

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

from Autoencoder import AutoEncoder, LZINBLoss, preprocess_data, HASGPU
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# 超参数
# BATCHSIZE = 32
EPS = 1e-9
# LR = 5e-3
# BETA1=0.99
# BETA2=0.999
# WEIGHT_DECAY = 1e-4
# EPOCH_NUM = 500
# EVER_SAVING = 100
FILE_NAME = 'data.csv'
# TURE_FILE = 'test_truedata.csv'
STATE_DICT_FILE = '0113epoch1000.pkl'
autoencoder = AutoEncoder(1000,hasBN=True)


lzinbloss = LZINBLoss(eps=EPS)
data = pd.read_csv(FILE_NAME).astype('float32')
data = torch.from_numpy(data.values.T)
# truedata = pd.read_csv(TURE_FILE).astype('float32')
# truedata = torch.from_numpy(truedata.values.T)

if HASGPU:
    autoencoder = autoencoder.cuda()
    lzinbloss = lzinbloss.cuda()
    data = data.cuda()



f = open('diary.txt', 'w')

if __name__ == '__main__':
    autoencoder.load_state_dict(torch.load(STATE_DICT_FILE))
    print(autoencoder)
    PI, M, THETA = autoencoder(data)
    iszero = data == 0
    predict_dropout_of_all = PI>0.5
    # dropout_predict = torch.where(predict_mask, M, torch.zeros_like(PI))
    # print("after",after)

    # true_drop_out_mask = iszero*((truedata - data)!=0)
    predict_dropout_mask = iszero*predict_dropout_of_all
    after = torch.floor(torch.where(predict_dropout_mask,M,data))
    zero_num = iszero.sum()
    # true_dropout_num = true_drop_out_mask.sum()
    predict_dropout_num = predict_dropout_mask.sum()
    print("predict_dropout_num:",predict_dropout_num,
        #   "\ntrue_dropout_num:", true_dropout_num,
          "\nzero_num:",zero_num)
        #   "\npredict out of true dropout rate:",(predict_dropout_mask*true_drop_out_mask).sum()/true_dropout_num)
    a = after.detach().numpy()
    print(len(a))
    np.savetxt('result.csv' , a.T, fmt='%.f', delimiter=',')
    # dif_after =  truedata - after
    # dif_true = truedata - data
    # print(dif_after)
    # print(dif_true)
    # print("predict distance:", torch.sqrt(torch.square(truedata - after).sum()).data,
        #   "origin distance:", torch.sqrt(torch.square(truedata - data).sum()).data)


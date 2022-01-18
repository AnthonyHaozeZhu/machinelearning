import numpy as np
import pandas as pd
import torch
import math
from Autoencoder import AutoEncoder, LZINBLoss


EPS = 1e-9
FILE_NAME = 'test_data.csv'
TURE_FILE = 'test_truedata.csv'
STATE_DICT_FILE = '0114epoch3000.pkl'
autoencoder = AutoEncoder(1000,hasBN=True)


lzinbloss = LZINBLoss(eps=EPS)
data = pd.read_csv(FILE_NAME).astype('float32')
data = torch.from_numpy(data.values.T)
len=math.floor(0.7*(len(data)))
data=data[len:]
truedata = pd.read_csv(TURE_FILE).astype('float32')
truedata = torch.from_numpy(truedata.values.T)
truedata=truedata[len:]






f = open('diary.txt', 'w')

if __name__ == '__main__':
    autoencoder.load_state_dict(torch.load(STATE_DICT_FILE))
    print(autoencoder)
    PI, M, THETA = autoencoder(data)
    iszero = data == 0
    predict_dropout_of_all = PI>0.5


    true_drop_out_mask = iszero*((truedata - data)!=0)
    predict_dropout_mask = iszero*predict_dropout_of_all
    after = torch.floor(torch.where(predict_dropout_mask,M,data))
    zero_num = iszero.sum()
    true_dropout_num = true_drop_out_mask.sum()
    predict_dropout_num = predict_dropout_mask.sum()
    predict_correct = (predict_dropout_mask*true_drop_out_mask).sum()
    recall=predict_correct/true_dropout_num
    precision = predict_correct/predict_dropout_num
    print("predict_dropout_num:",predict_dropout_num,
          "\ntrue_dropout_num:", true_dropout_num,
          "\nzero_num:",zero_num,
          "\npredict_correct:",predict_correct,
          "\npredict out of true dropout rate:",(predict_dropout_mask*true_drop_out_mask).sum()/true_dropout_num,
          "\nprecision:",precision,
          "\nrecall:",recall)

    dif_after =  truedata - after
    dif_true = truedata - data
    # print(dif_after)
    # print(dif_true)
    print("predict distance:", torch.sqrt(torch.square(truedata - after).sum()).data,
          "origin distance:", torch.sqrt(torch.square(truedata - data).sum()).data)


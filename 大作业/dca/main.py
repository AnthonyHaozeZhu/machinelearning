import os.path

import torch, argparse
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import csv
import numpy as np

class FullConnectBasicBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullConnectBasicBlock, self).__init__()
        self.fulc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fulc(x))
        return x


class DCA(nn.Module):
    def __init__(self, input_size):
        super(DCA, self).__init__()
        self.net = nn.Sequential(*[
            FullConnectBasicBlock(input_size, 64),
            FullConnectBasicBlock(64, 32),
            FullConnectBasicBlock(32, 64),
        ])
        self.mat_avg = nn.Linear(64, input_size)
        self.mat_pai = nn.Linear(64, input_size)
        self.mat_theta = nn.Linear(64, input_size)

    def forward(self, x):
        x = self.net(x)
        m = torch.exp(self.mat_avg(x))
        p = torch.sigmoid(self.mat_pai(x))
        t = torch.exp(self.mat_theta(x))
        return m, p, t

def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = torch.sum(~torch.isnan(x))
    return torch.where(torch.equal(nelem, 0.), 1., nelem)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return torch.divide(torch.sum(x), nelem)


def mse_loss(y_true, y_pred):
    ret = torch.square(y_pred - y_true)

    return _reduce_mean(ret)

class NB(object):
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False):

        # for numerical stability
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        y_true = y_true
        y_pred = y_pred * scale_factor

        if self.masking:
            nelem = _nelem(y_true)
            y_true = _nan2zero(y_true)

        # Clip theta
        theta = min(self.theta, 1e6)

        t1 = torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
        t2 = (theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))
        final = t1 + t2

        final = _nan2inf(final)

        if mean:
            if self.masking:
                final = torch.divide(torch.sum(final), nelem)
            else:
                final = torch.mean(final)


        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps


        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0-self.pi+eps)

        y_pred = y_pred * scale_factor
        theta = min(self.theta, 1e6)

        zero_nb = torch.pow(theta/(theta+y_pred+eps), theta)
        zero_case = -torch.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
        result = torch.where(torch.less(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge

        if mean:
            if self.masking:
                result = _reduce_mean(result)
            else:
                result = torch.mean(result)

        result = _nan2inf(result)
        return result

def getZINBLoss(X, P, M, T):
    loss = 0
    for batch in range(X.shape[0]):
        for x, p, m, t in zip(X[batch], P[batch], M[batch], T[batch]):
            loss_func = ZINB(pi=p, theta=t)
            loss += loss_func.loss(x, m)

    return loss / X.shape[0]


class CellDataset(Dataset):
    def __init__(self, csv_path):
        super(CellDataset, self).__init__()
        self.data = []
        with open(csv_path, "r") as f:
            raw_data = csv.reader(f)
            for d in raw_data:
                if len(d) != len(self.data):
                    self.data = [[] for _ in range(len(d))]
                for i, t in enumerate(d):
                    if "Cell" in t:
                        continue
                    self.data[i].append(int(t))
        self.data = np.array(self.data, dtype=np.float32)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d = self.data[item]
        d /= len(d)
        d += 1
        d = np.log2(d)
        d /= np.max(d) - np.min(d)

        return torch.tensor(d)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data.csv", help="The path of training dataset. Default: ./data.csv")
    parser.add_argument("--val_path", default="./test_data.csv", help="The path of validation dataset. Default: ./test_data.csv")
    parser.add_argument("--valtrue_path", default="./test_truedata.csv", help="The path of validation ground truth. Default: ./test_truedata.csv")
    parser.add_argument("--batchsize", default=100)
    parser.add_argument("--epoch", default=50)
    return vars(parser.parse_args())


def main(args):
    model = DCA(input_size=1000)
    print("Preparing for dataset...")
    dataset = CellDataset(args["train_path"])
    val_dataset = CellDataset(args["val_path"])
    val_dataset_true = CellDataset(args["valtrue_path"])
    val_dataloader = DataLoader(dataset = val_dataset, batch_size=int(args["batchsize"]), shuffle=False, num_workers=8)
    val_dataloader_true = DataLoader(dataset = val_dataset_true, batch_size=int(args["batchsize"]), shuffle=False, num_workers=8)
    dataloader = DataLoader(dataset = dataset, batch_size=int(args["batchsize"]), shuffle=True, num_workers=8)
    optim = torch.optim.Adam(model.parameters())
    mseloss = nn.MSELoss()
    print(model)
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")
    for epoch in range(int(args["epoch"])):
        model.train()
        for data in dataloader:
            optim.zero_grad()
            m, p, t = model(data)
            loss = getZINBLoss(data, p, m, t)
            loss.backward()
            optim.step()
            print("Epoch:", epoch, "/", args["epoch"], "Loss:", loss.detach().numpy())

        model.eval()
        print("====> validating...")
        with torch.no_grad():
            mse = 0
            all = 0
            dropout = 0
            for test, gt in zip(val_dataloader, val_dataloader_true):
                m, _, _ = model(test)
                mse += mseloss(m, gt)
                for n in range(m.shape[0]):
                    for t, g, p in zip(test[n], gt[n], m[n]):
                        all += 1
                        if t == 0 and g != 0 and p == 0:
                            dropout += 1
            mse /= len(val_dataloader)
            print("MSE:", mse.detach().numpy())
            print("Dropout:", dropout / all)
        torch.save(model, "./checkpoint/Epoch{}.pkl".format(epoch))


if __name__ == '__main__':
    main(get_args())
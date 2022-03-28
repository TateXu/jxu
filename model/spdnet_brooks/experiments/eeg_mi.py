
import sys
sys.path.append('/home/jxu/anaconda3/lib/python3.7/site-packages/jxu/model/spdnet_brooks/')

from spd.optimizers import MixOptimizer
import cplx.nn as nn_cplx
import spd.nn as nn_spd
from torch.utils import data
import torch.nn as nn
import torch as th
import numpy as np
import random
import os
from pathlib import Path
import sys

from moabb.datasets import MunichMI, BNCI2014001
from moabb.paradigms import MotorImagery
# from moabb.pipelines.features import TSSF

sys.path.insert(0, '..')
device = 'cpu'

"""
cpu time vs cuda time for current version upper cpu and lower cuda
(Pdb) print(train_mean)
  zero_grad,         model,         loss_fn,       acc_cal,       bp,             step
[3.59808074e-04 9.61264345e-02 1.26359198e-04 5.19328647e-05 9.83133581e-03 1.72788019e-01]
[8.53263007e-04 2.03452839e+00 1.43705474e-04 1.03863080e-04 1.13091363e-02 6.75106499e-01]


(Pdb) print(test_mean)
   model_test,        loss_fn,       label,         acc
[2.94072178e-02 1.56437026e-04 3.51349513e-05 3.38898765e-05]
[6.15100752e-01 1.81497468e-04 7.07017051e-05 7.16951158e-05]

"""
def hdm05(data_loader):

    # main parameters
    lr = 5e-2
    n = 93  # dimension of the data
    C = 117  # number of classes
    threshold_reeig = 1e-4  # threshold for ReEig layer
    epochs = 200

    if not Path(data_path).is_dir():
        print("Error: dataset not found")
        print("Please download and extract the file at the following url: http://www-connex.lip6.fr/~schwander/datasets/hdm05.tgz")
        sys.exit(2)

    # setup data and model
    class HDMNet(nn.Module):
        def __init__(self, dim_list=[128, 128, 10, 10, 10, 10]):
            super(__class__, self).__init__()
            dim0 = 128
            dim1 = 128
            dim2 = 10
            dim3 = 10
            dim4 = 10
            dim5 = 10

            self.dim_list = dim_list
            classes = 2
            # ch_out, ch_in, dim_in, dim_out

            self.batchnorm_list, self.bimap_list, self.reeig_list, self.logeig_list = [] * 4
            self.batchnorm_list.append(nn_spd.BatchNormSPD(dim_list[0]))
            for layer_ind, layer_dim in enumerate(dim_list[1:]):
                self.bimap_list.append(
                    nn_spd.BiMap(1, 1, layer_dim, dim_list[layer_ind+1]))
                self.batchnorm_list.append(nn_spd.BatchNormSPD(layer_dim))
                self.reeig_list.append(nn_spd.ReEig())
                self.logeig_list.append(nn_spd.LogEig())

            self.bimap1 = nn_spd.BiMap(1, 1, dim0, dim1)
            self.batchnorm1 = nn_spd.BatchNormSPD(dim1)
            self.reeig1 = nn_spd.ReEig()
            self.logeig1 = nn_spd.LogEig()

            self.bimap2 = nn_spd.BiMap(1, 1, dim1, dim2)
            self.batchnorm2 = nn_spd.BatchNormSPD(dim2)
            self.reeig2 = nn_spd.ReEig()
            self.logeig2 = nn_spd.LogEig()

            self.bimap3 = nn_spd.BiMap(1, 1, dim2, dim3)
            self.batchnorm3 = nn_spd.BatchNormSPD(dim3)
            self.reeig3 = nn_spd.ReEig()
            self.logeig3 = nn_spd.LogEig()

            self.bimap4 = nn_spd.BiMap(1, 1, dim3, dim4)
            self.batchnorm4 = nn_spd.BatchNormSPD(dim4)
            self.reeig4 = nn_spd.ReEig()
            self.logeig4 = nn_spd.LogEig()


            self.bimap5 = nn_spd.BiMap(1, 1, dim4, dim5)
            self.batchnorm5 = nn_spd.BatchNormSPD(dim5)
            self.reeig5 = nn_spd.ReEig()
            self.logeig5 = nn_spd.LogEig()


            self.linear = nn.Linear(dim3**2, classes).to(device).double()

        def forward(self, x):
            # x_spd = self.bimap1(x)
            # x_vec = self.logeig(self.reeig(x_spd)).view(x_spd.shape[0], -1)

            # x_spd1 = self.logeig1(self.reeig1(self.batchnorm1(self.bimap1(x))))
            # x_spd2 = self.logeig2(self.reeig2(self.batchnorm2(self.bimap2(x_spd1))))
            # x_spd3 = self.logeig3(self.reeig3(self.batchnorm3(self.bimap3(x_spd2))))
            x_spd1 = self.reeig1(self.batchnorm1(self.bimap1(self.batchnorm0(x))))
            # x_spd1 = self.reeig1(self.batchnorm1(self.bimap1(x)))
            x_spd2 = self.reeig2(self.batchnorm2(self.bimap2(x_spd1)))

            # x_spd3 = self.logeig3(self.reeig3(self.batchnorm3(self.bimap3(x_spd2))))
            # x_vec = x_spd3.view(x_spd3.shape[0], -1)

            x_spd3 = self.reeig3(self.batchnorm3(self.bimap3(x_spd2)))
            x_spd4 = self.reeig4(self.batchnorm4(self.bimap4(x_spd3)))
            x_spd5 = self.logeig5(self.reeig5(self.batchnorm5(self.bimap5(x_spd4))))
            x_vec = x_spd5.view(x_spd5.shape[0], -1)
            y = self.linear(x_vec)
            return y
    model = HDMNet()

    # setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opti = MixOptimizer(model.parameters(), lr=lr)

    # initial validation accuracy
    loss_val, acc_val = [], []
    y_true, y_pred = [], []
    gen = data_loader._test_generator
    model.eval()
    for local_batch, local_labels in gen:
        out = model(local_batch)
        l = loss_fn(out, local_labels)
        predicted_labels = out.argmax(1)
        y_true.extend(list(local_labels.cpu().numpy()))
        y_pred.extend(list(predicted_labels.cpu().numpy()))
        acc, loss = (predicted_labels == local_labels).cpu(
        ).numpy().sum()/out.shape[0], l.cpu().data.numpy()
        loss_val.append(loss)
        acc_val.append(acc)
    acc_val = np.asarray(acc_val).mean()
    loss_val = np.asarray(loss_val).mean()
    print('Initial validation accuracy: '+str(100*acc_val)+'%')

    # training loop
    for epoch in range(epochs):

        # train one epoch
        loss_train, acc_train = [], []
        model.train()
        for local_batch, local_labels in data_loader._train_generator:
            opti.zero_grad()
            out = model(local_batch)
            l = loss_fn(out, local_labels)
            acc, loss = (out.argmax(1) == local_labels).cpu(
            ).numpy().sum()/out.shape[0], l.cpu().data.numpy()
            loss_train.append(loss)
            acc_train.append(acc)
            l.backward()
            opti.step()
        acc_train = np.asarray(acc_train).mean()
        loss_train = np.asarray(loss_train).mean()

        # validation
        acc_val_list = []
        y_true, y_pred = [], []
        model.eval()
        for local_batch, local_labels in data_loader._test_generator:
            out = model(local_batch)
            l = loss_fn(out, local_labels)
            predicted_labels = out.argmax(1)
            y_true.extend(list(local_labels.cpu().numpy()))
            y_pred.extend(list(predicted_labels.cpu().numpy()))
            acc, loss = (predicted_labels == local_labels).cpu(
            ).numpy().sum()/out.shape[0], l.cpu().data.numpy()
            acc_val_list.append(acc)
        xx = None
        acc_val = np.asarray(acc_val_list).mean()
        print('Val acc: ' + str(100*acc_val) + '% at epoch ' +
              str(epoch + 1) + '/' + str(epochs))

    print('Final validation accuracy: '+str(100*acc_val)+'%')
    return 100*acc_val


if __name__ == "__main__":

    data_path = '/home/jxu/File/Data/SPDNet/data/hdm05/'
    # data_path = 'data/hdm05/'
    class DatasetMOABB(data.Dataset):
        def __init__(self, data, trial_idx=None):
            self._trial_idx = trial_idx
            self.data = data
            self.data2set()

        def __len__(self):
            # returns the number of samples in the dataset
            return len(self._trial_idx)

        def __getitem__(self, idx):
            # loads and returns a sample from the dataset at the given index
            # item.
            return self._X[idx].to(device), self._y[idx].to(device)

        def data2set(self):
            self._X, self._y = self.data[0], self.data[1]
            if self._trial_idx is not None:
                self._X, self._y = self._X[self._trial_idx], self._y[self._trial_idx]

            self._X = self._X[:, None, :, :]

    class DataLoaderMOABB:
        def __init__(self, subject=1, session=0, fmin=8, fmax=32,
                     paradigms=None, events=['left_hand', 'right_hand'],
                     pval=0.5, batch_size=10, cov='scm'):
            self.subject = subject
            self.session = session
            self.fmin = fmin
            self.fmax = fmax
            self.paradigms = paradigms
            self.events = events
            self.pval = pval
            self.cov = cov

            self.preload_data()  # Load X and y from MOABB dataset
            train_set = DatasetMOABB(data=[self.X_all, self.y_all],
                                     trial_idx=self._train_ind)
            test_set = DatasetMOABB(data=[self.X_all, self.y_all],
                                    trial_idx=self._test_ind)
            self._train_generator = data.DataLoader(
                train_set, batch_size=batch_size, shuffle='True')
            self._test_generator = data.DataLoader(
                test_set, batch_size=batch_size, shuffle='True')
            import pdb;pdb.set_trace()

        def preload_data(self):

            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split

            MI = MotorImagery(fmin=self.fmin, fmax=self.fmax,
                              events=self.events)
            dataset = MunichMI()  # BNCI2014001()
            X, y, metadata = MI.get_data(dataset, [self.subject])
            for nr_ses, name_ses in enumerate(np.unique(metadata.session).tolist()):
                if nr_ses == self.session:
                    ix = metadata.session == name_ses
                    X_all, y_all = X[ix], y[ix]

            if self.cov:
                from pyriemann.estimation import Covariances
                self.X_all = th.from_numpy(
                    Covariances(estimator=self.cov).fit_transform(X_all)).double()
            else:
                self.X_all = th.from_numpy(X_all).double()
            self.y_all = th.from_numpy(
                LabelEncoder().fit(np.unique(y_all)).transform(y_all)).long()

            self._train_ind, self._test_ind, _, _ = train_test_split(
                np.asarray(range(len(self.y_all))), self.y_all,
                test_size=self.pval, stratify=self.y_all)

    pval = 0.2   # test percentage
    batch_size = 10  # batch size
    aa = DataLoaderMOABB(subject=2, session=0, pval=pval,
                         batch_size=batch_size)
    hdm05(aa)
    import pdb;pdb.set_trace()



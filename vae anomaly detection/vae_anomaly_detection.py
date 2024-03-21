import torch
import torch.nn as  nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader , Dataset
import torchvision as transforms
import os , sys
import dotenv
import pytorch_lightning as pl
from sklearn import preprocessing
from pytorch_lightning.loggers import WandbLogger
from argparse import Namespace
import sklearn
from  pathlib import Path
from collections import OrderedDict
#import wandb
from sklearn.preprocessing import LabelEncoder

for dirname, _ , filenames in os.walk('data/'):
    for file in filenames:
        print(os.path.join(dirname,file))

data_path = 'data/realKnownCause/machine_temperature_system_failure.csv'
data_root = 'data'

'''

We do not have much information about what kind of machine or industry we are dealing with, which is a bit of an issue when trying to use this data, especially when it comes to feature engineering. We will therefore have to make assumptions.

The timestamps cover the Christmas and New Year holidays. Since we are dealing with an industrial machine, it stands to reason that its workload might be affected by holidays, and maybe even by the proximity (in time) of a holiday. In the absence of additional information, we are going to assume that the applicable holidays are those typical in Europe and the Americas, i.e. Christmas and New Year's Day. By the same reasoning, we might need to know the day of the week (possibly lower workload on weekends?) or the hour of the day. Again, we will assume that weekends are Satuday and Sunday.
'''

class DataModuleProcess(pl.LightningDataModule):
    def __init__(self,batch_size,data_path,data_root):
        super().__init__()
        self.data_path = data_path
        self.data_root = data_root
    def loading_data(self):
        
        df = pd.read_csv(self.data_path)

        data = df.copy()
        raw_value = data['value']
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['day'] = data['timestamp'].dt.day
        data['hour_min'] = data['timestamp'].dt.hour + data['timestamp'].dt.minute/60
        data['month'] = data['timestamp'].dt.month
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['holiday'] = 0
        data.loc[(data['day'] == 25) & (data['month'] == 12),'holiday'] = 1  # Christmas
        data.loc[(data['day'] == 1) & (data['month'] == 1),'holiday'] = 1  # New Year's Day
        holidays = data.loc[data['holiday'] == 1, 'timestamp'].dt.date.unique()
        print(holidays)
        for i, hd in enumerate(holidays):
            data['hol_' + str(i)] = data['timestamp'].dt.date - hd

        print(data.head())
        for i in range(data.shape[0]):
            if np.abs(data.loc[data.index[i], 'hol_0']) <= np.abs(data.loc[data.index[i], 'hol_1']):
                data.loc[data.index[i], 'gap_holiday'] = data.loc[data.index[i], 'hol_0']
            else:
                data.loc[data.index[i], 'gap_holiday'] = data.loc[data.index[i], 'hol_1']
        data['gap_holiday'] = data['gap_holiday'].astype('timedelta64[ns]')

        print(data.head())
        data.drop(['hol_0', 'hol_1'], axis=1, inplace=True)
        print(data.head())
        data['t'] = (data['timestamp'].astype(np.int64)/1e11).astype(np.int64)
        data.drop('timestamp', axis=1, inplace=True)

        cont_vars = ['value', 'hour_min', 'gap_holiday', 't']
        cat_vars = ['day', 'month', 'day_of_week', 'holiday']
        return cont_vars, cat_vars , data
    def labelencoding(self):
        cont_vars, cat_vars , data = DataModuleProcess.loading_data(self)
        label_encoders = [LabelEncoder() for _ in cat_vars]
        for col, enc in zip(cat_vars, label_encoders):
            data[col] = enc.fit_transform(data[col])
        test_ratio = 0.3
        tr_data = data.iloc[: int(len(data) * (1 - test_ratio))]
        tst_data = data.iloc[int(len(data) * (1 - test_ratio)) :]

        print(tr_data.head())
        return tr_data , tst_data
    def ScalingData(self):

        cont_vars, cat_vars , data = DataModuleProcess.loading_data(self)
        tr_data , tst_data= DataModuleProcess.labelencoding(self) 
        scaler = preprocessing.StandardScaler().fit(tr_data[cont_vars])
        tr_data_scaled = tr_data.copy()
        tr_data_scaled[cont_vars] = scaler.transform(tr_data[cont_vars])
        tst_data_scaled = tst_data.copy()
        tst_data_scaled[cont_vars] = scaler.transform(tst_data[cont_vars])
        return tr_data_scaled





class Encoder(nn.Module):

    def __init__(self, **hparams):
        super().__init__()

        self.hparams = Namespace(**hparams)
        self.fc1 = nn.Linear(self.hparams.input_dim ,self.hparams.h_dim )
        self.fc2 = nn.Linear(self.hparams.h_dim , self.hparams.h_dim2)
        self.fc3  = nn.Linear(self.hparams.h_dim2, self.hparams.h_dim3)

        self.mu = nn.Linear(self.hparams.h_dim3, self.hparams.z_dim)
        self.log_var = nn.Linear(self.hparams.h_dim3, self.hparams.z_dim)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        mu = self.mu(x)
        log_var = self.log_var(x)
        sigma = torch.randn(mu.shape[0],1)


        return mu, log_var ,x

class Decoder(nn.Module):

    def __init__(self, **hparams):
        super().__init__()

        self.hparams = Namespace(**hparams)
        self.fc1 = nn.Linear(self.hparams.z_dim ,self.hparams.h_dim3 )
        self.fc2 = nn.Linear(self.hparams.h_dim3 , self.hparams.h_dim2)
        self.fc3  = nn.Linear(self.hparams.h_dim2, self.hparams.h_dim1)

        self.ouput = nn.Linear(self.hparams.h_dim1, self.hparams.input_dim)
        self.relu = nn.ReLU()
    def forward(self,z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = self.relu(self.fc3(z))
        out = self.ouput(z)
        return out


class VariationalEncoders(pl.LightningModule):
    def __init__(self,**hparams):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)
    def reparametrize(self,mu,log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * self.hparams.stdev
        z = eps * std + mu
        return z
    def forward(self,x):
        mu , log_var ,x = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        output = self.decoder(z)
        return output , mu,  log_var  , x
    def lossfn(self,out , mu , log_var , x):
        
if __name__ == '__main__':

    obj = DataModuleProcess(data_path=data_path,data_root=data_root,batch_size=32)
    train_data = obj.ScalingData()

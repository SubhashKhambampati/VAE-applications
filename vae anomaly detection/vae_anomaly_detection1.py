from matplotlib._api import rename_parameter
from sqlalchemy.types import Variant
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
datasets_root = Path('data')

'''

We do not have much information about what kind of machine or industry we are dealing with, which is a bit of an issue when trying to use this data, especially when it comes to feature engineering. We will therefore have to make assumptions.

The timestamps cover the Christmas and New Year holidays. Since we are dealing with an industrial machine, it stands to reason that its workload might be affected by holidays, and maybe even by the proximity (in time) of a holiday. In the absence of additional information, we are going to assume that the applicable holidays are those typical in Europe and the Americas, i.e. Christmas and New Year's Day. By the same reasoning, we might need to know the day of the week (possibly lower workload on weekends?) or the hour of the day. Again, we will assume that weekends are Satuday and Sunday.
'''

df = pd.read_csv(data_path)
data = df.copy()
raw_value = data['value']
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['day'] = data['timestamp'].dt.day
data['hour_min'] = data['timestamp'].dt.hour + data['timestamp'].dt.minute/60
data['month'] = data['timestamp'].dt.month
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['holiday'] = 0
print(data.head())
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
print(data.head())
label_encoders = [LabelEncoder() for _ in cat_vars]
for col, enc in zip(cat_vars, label_encoders):
    data[col] = enc.fit_transform(data[col])


data['gap_holiday'] = data['gap_holiday'].dt.total_seconds().astype(float)
test_ratio = 0.3
tr_data = data.iloc[: int(len(data) * (1 - test_ratio))]
tst_data = data.iloc[int(len(data) * (1 - test_ratio)) :]

print(tr_data.dtypes)
print(tr_data.head())
scaler = preprocessing.StandardScaler().fit(tr_data[cont_vars])
tr_data_scaled = tr_data.copy()
tr_data_scaled[cont_vars] = scaler.transform(tr_data[cont_vars])
tst_data_scaled = tst_data.copy()
tst_data_scaled[cont_vars] = scaler.transform(tst_data[cont_vars])
tst_data_scaled = pd.DataFrame(tst_data_scaled)
print(tr_data_scaled.head())

tr_data_scaled.to_csv(datasets_root/'train.csv', index=False)
tst_data_scaled.to_csv(datasets_root/'test.csv', index=False)


class TSDataset(Dataset):
    def __init__(self, split, cont_vars=None, cat_vars=None, lbl_as_feat=True):
        """
        split: 'train' if we want to get data from the training examples, 'test' for
        test examples, or 'both' to merge the training and test sets and return samples
        from either.
        cont_vars: List of continuous variables to return as features. If None, returns
        all continuous variables available.
        cat_vars: Same as above, but for categorical variables.
        lbl_as_feat: Set to True when training a VAE -- the labels (temperature values)
        will be included as another dimension of the data. Set to False when training
        a model to predict temperatures.
        """
        super().__init__()
        assert split in ['train', 'test', 'both']
        self.lbl_as_feat = lbl_as_feat
        if split == 'train':
            self.df = pd.read_csv(datasets_root/'train.csv')
        elif split == 'test':
            self.df = pd.read_csv(datasets_root/'test.csv')
        else:
            df1 = pd.read_csv(datasets_root/'train.csv')
            df2 = pd.read_csv(datasets_root/'test.csv')
            self.df = pd.concat((df1, df2), ignore_index=True)
        
        # Select continuous variables to use
        if cont_vars:
            self.cont_vars = cont_vars
            # If we want to use 'value' as a feature, ensure it is returned
            if self.lbl_as_feat:
                try:
                    assert 'value' in self.cont_vars
                except AssertionError:
                    self.cont_vars.insert(0, 'value')
            # If not, ensure it not returned as a feature
            else:
                try:
                    assert 'value' not in self.cont_vars
                except AssertionError:
                    self.cont_vars.remove('value')
                    
        else:  # if no list provided, use all available
            self.cont_vars = ['value', 'hour_min', 'gap_holiday', 't']
        
        # Select categorical variables to use
        if cat_vars:
            self.cat_vars = cat_vars
        else:  # if no list provided, use all available
            self.cat_vars = ['day', 'month', 'day_of_week', 'holiday']
        
        # Finally, make two Numpy arrays for continuous and categorical
        # variables, respectively:
        if self.lbl_as_feat:
            self.cont = self.df[self.cont_vars].copy().to_numpy(dtype=np.float32)
        else:
            self.cont = self.df[self.cont_vars].copy().to_numpy(dtype=np.float32)
            self.lbl = self.df['value'].copy().to_numpy(dtype=np.float32)
        self.cat = self.df[self.cat_vars].copy().to_numpy(dtype=np.int64)
            
    def __getitem__(self, idx):
        if self.lbl_as_feat:  # for VAE training
            return torch.tensor(self.cont[idx]), torch.tensor(self.cat[idx])
        else:  # for supervised prediction
            return torch.tensor(self.cont[idx]), torch.tensor(self.cat[idx]), torch.tensor(self.lbl[idx])
    
    def __len__(self):
        return self.df.shape[0]


ds = TSDataset(split='both', cont_vars=['value', 't'], cat_vars=['day_of_week', 'holiday'], lbl_as_feat=True)
print(len(ds))
it = iter(ds)
for _ in range(10):
    print(next(it))







class Encoder(nn.Module):

    def __init__(self, **hparams):
        super().__init__()

        self.hparams = Namespace(**hparams)
        self.fc1 = nn.Linear(self.hparams.input_dim ,self.hparams.h_dim1 )
        self.fc2 = nn.Linear(self.hparams.h_dim1 , self.hparams.h_dim2)
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
        out = torch.sigmoid(out)
        return out


class VariationalEncoders(nn.Module):
    def __init__(self,**hparams):
        super().__init__()
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)
    def reparametrize(self,mu,log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) * self.hparams.stdev1
        z = eps * std + mu
        return z
    def forward(self,batch):
        x  = batch
        mu , log_var , data_h = self.encoder(x)
        z = self.reparametrize(mu,log_var)
        out = self.decoder(z)
        return out , mu , log_var,x
def vae_loss(mu,log_var,out,x):
    recon_loss = torch.nn.functional.smooth_l1_loss(out,x,reduction='mean')
    kld= -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp())

    loss = recon_loss + 0.05 * kld
    return recon_loss, kld,loss



cont_features = ['value', 'hour_min', 'gap_holiday', 't'] 
cat_features = ['day_of_week', 'holiday']  # Remember that we removed `day` and `month`
hparams = OrderedDict(
    cont_vars = cont_features,
    cat_vars = cat_features,
    z_dim = 16,
    stdev1 = 0.1,
    kld_beta = 0.05,
    lr = 0.001,
    weight_decay = 1e-5,
    batch_size = 128,
    epochs = 60,
    input_dim = 6,
    h_dim1 = 10,
    h_dim2 = 20,
    h_dim3 = 30,
    
)


model = VariationalEncoders(**hparams)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

dataset = TSDataset('train', cont_vars=cont_vars, 
            cat_vars = cat_vars, lbl_as_feat=True
        )
    
num_epochs = 10
device = 'cpu'
train_loader = DataLoader(dataset,shuffle=True,batch_size = 32)
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (x) in enumerate(train_loader):
        optimizer.zero_grad()
        print(len(x))
        x_recon, mu, log_var = model(x)
        recon_loss,kld,loss = vae_loss(x_recon, x, mu, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

# Anomaly detection (Calculate reconstruction error)
model.eval()
reconstruction_errors = []
with torch.no_grad():
    for x  in train_loader:

        print(len(x))
        x_recon, mu, log_var = model(x)
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='none')
        reconstruction_errors.extend(recon_loss.sum(dim=1).cpu().numpy())

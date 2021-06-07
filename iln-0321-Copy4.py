#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dataclasses
from dataclasses import dataclass


# In[2]:


import pandas as pd
import numpy as np
import pickle
import os
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from datetime import date, timedelta, datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# In[3]:


gpu = '5'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


checkpoint_path = './pth'

torch.set_num_threads(3)


# In[4]:


from IPython.display import display, HTML
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)
pd.set_option('display.precision', 20)
display(HTML("<style>.container { width:90% !important; }</style>")) 


# In[5]:


np.random.seed(1)

class CFG:
    batch_size=64
    n_time = 3
    total_n_bssid = 60
    n_bssid = 40
    n_tar = 1
    ibeacon_seq_len = 20
    n_sensor = 6
    n_sensor_feature = 13


# ### data

# In[6]:


with open('dic_data_v1.pickle', 'rb') as f:
    dic_data = pickle.load(f)


# In[8]:


get_ipython().run_cell_magic('time', '', "import math\nfor uid, dic_uid in dic_data.items():\n    l_idx, l_idx_sub = [], []\n    len_time = len(dic_uid['time'])\n#     len_time = int((dic_uid['y'][-1, 0] - dic_uid['y'][0, 0])/1e-7) + 2\n\n    t_int = np.linspace(dic_uid['y'][0, 0], dic_uid['y'][-1, 0], len_time)\n    y_int = np.zeros([len_time, 4])\n    y_int[:, 0] = t_int\n    y_int[:, 1] = dic_uid['y'][0, 1]\n    y_int[:, 2] = np.interp(t_int, dic_uid['y'][:, 0], dic_uid['y'][:, 2])\n    y_int[:, 3] = np.interp(t_int, dic_uid['y'][:, 0], dic_uid['y'][:, 3])    \n    dic_uid['y_int'] = y_int    \n\n    for tar, time_tar in enumerate(dic_uid['y_int'][:, 0]):\n        time_tar = t_int[tar]\n        s_wifi, e_wifi = get_se(dic_uid['time'], time_tar, CFG.n_time)\n        l_idx.append((uid, tar, s_wifi, e_wifi))\n\n    for tar, time_tar in enumerate(dic_uid['y'][:, 0]):\n        s_wifi, e_wifi = get_se(dic_uid['time'], time_tar, CFG.n_time)\n        l_idx_sub.append((uid, tar, s_wifi, e_wifi))\n            \n    dic_uid['l_idx'] = l_idx\n    dic_uid['l_idx_sub'] = l_idx_sub")


# In[13]:


ar_uid = np.array(list(dic_data.keys()))
l_site = []
for uid in ar_uid:
    l_site.append(dic_data[uid]['site'])

from sklearn.model_selection import StratifiedKFold
stk = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

for idx_uid_train, idx_uid_valid in stk.split(ar_uid, l_site):
    break

l_idx_train, l_idx_valid = [], []

for uid in ar_uid[idx_uid_train]:
    l_idx_train += dic_data[uid]['l_idx']
for uid in ar_uid[idx_uid_valid]:
    l_idx_valid += dic_data[uid]['l_idx_sub']


# In[14]:


len(l_idx_train), len(l_idx_valid) # 이전 (67638, 7568)


# In[15]:


l_idx_train[:3]


# ### DataLoader

# In[16]:


def get_idx(s, e, is_test):
    if is_test:
        N = CFG.n_bssid
    else:
        N = CFG.total_n_bssid
        
    l = []
    for i in range(s, e):
        for j in range(N):
            l.append((i, j))
            
    if is_test:
        ar = np.array(l)
    else:          
        ar = np.array(l)[np.random.choice(range(len(l)), (e-s)*CFG.n_bssid, replace=False)]
    
    return ar[:, 0], ar[:, 1]


# In[17]:


max_time = 11000

class ILNDataset(Dataset):
    def __init__(self, l_idx, test=False):
        self.l_idx = l_idx
        self.test = test
        
    def __getitem__(self, idx):
        u, t, s, e = self.l_idx[idx]  
        dic_ = dic_data[u]

        bssid = np.zeros([CFG.n_time * CFG.n_bssid])
        rssi = np.zeros([CFG.n_time * CFG.n_bssid])
        resp = np.zeros([CFG.n_time * CFG.n_bssid])
        time = np.zeros([CFG.n_time * CFG.n_bssid])
        lst = np.zeros([CFG.n_time * CFG.n_bssid])
        d_time = np.zeros([CFG.n_time * CFG.n_bssid])
        d_lst = np.zeros([CFG.n_time * CFG.n_bssid])

        if self.test:
            y = dic_['y'][t]            
            idx_bssid = range(CFG.n_bssid)
        else:
            y = dic_['y_int'][t]
            idx_bssid = np.random.choice(range(CFG.total_n_bssid), CFG.n_bssid, replace=False)
            
        length = (e-s) * CFG.n_bssid
        bssid[:length] = dic_['bssid'][s:e, idx_bssid].flatten()
        rssi[:length] = dic_['rssi'][s:e, idx_bssid].flatten()
        resp[:length] = dic_['resp'][s:e, idx_bssid].flatten()
        time[:length] = dic_['time'][s:e].repeat(CFG.n_bssid)
        lst[:length] = dic_['lst'][s:e, idx_bssid].flatten()
        d_time[:length] = dic_['time'][s:e].repeat(CFG.n_bssid) - y[0]
        d_lst[:length] = dic_['lst'][s:e, idx_bssid].flatten() - y[0]
        
        mask_t = np.expand_dims(bssid!=0, -2) * 1
        mask_b = bssid==0
#         subsequent_mask = np.triu(np.ones((CFG.n_time, CFG.n_time)), k=1).astype('uint8') == 0  
       
        return bssid.astype(np.int64), rssi.astype(np.int64), resp.astype(np.float64), time.astype(np.float64), lst.astype(np.float64),     d_time.astype(np.float64), d_lst.astype(np.float64), y.astype(np.float64), mask_b, mask_t

    
    def __len__(self):
        return len(self.l_idx)
    

train_db = ILNDataset(l_idx_train)
valid_db = ILNDataset(l_idx_valid, test=True)

train_loader = DataLoader(train_db, batch_size=CFG.batch_size, num_workers=1, shuffle=True)
valid_loader = DataLoader(valid_db, batch_size=CFG.batch_size, num_workers=1, shuffle=False)


# In[18]:


max_time = 11000

class ILNDataset(Dataset):
    def __init__(self, l_idx, test=False):
        self.l_idx = l_idx
        self.test = test
        
    def __getitem__(self, idx):
        u, t, s, e = self.l_idx[idx]  
        dic_ = dic_data[u]

        bssid = np.zeros([CFG.n_time * CFG.n_bssid])
        rssi = np.zeros([CFG.n_time * CFG.n_bssid])
        resp = np.zeros([CFG.n_time * CFG.n_bssid])
        time = np.zeros([CFG.n_time * CFG.n_bssid])
        lst = np.zeros([CFG.n_time * CFG.n_bssid])
        d_time = np.zeros([CFG.n_time * CFG.n_bssid])
        d_lst = np.zeros([CFG.n_time * CFG.n_bssid])

        if self.test:
            y = dic_['y'][t]            
            idx_bssid = range(CFG.n_bssid)
            
            length = (e-s) * CFG.n_bssid
            bssid[:length] = dic_['bssid'][s:e, idx_bssid].flatten()
            rssi[:length] = dic_['rssi'][s:e, idx_bssid].flatten()
            resp[:length] = dic_['resp'][s:e, idx_bssid].flatten()
            time[:length] = dic_['time'][s:e].repeat(CFG.n_bssid)
            lst[:length] = dic_['lst'][s:e, idx_bssid].flatten()
            d_time[:length] = dic_['time'][s:e].repeat(CFG.n_bssid) - y[0]
            d_lst[:length] = dic_['lst'][s:e, idx_bssid].flatten() - y[0]    
            
        else:
            y = dic_['y_int'][t]
#             idx_bssid = np.random.choice(range(CFG.total_n_bssid), CFG.n_bssid, replace=False)
            idx_time = np.random.choice(range(s, e), CFG.n_time * CFG.n_bssid, replace=True)
            idx_bssid = np.random.choice(range(CFG.total_n_bssid), CFG.n_time * CFG.n_bssid, replace=True)
            
            bssid = dic_['bssid'][idx_time, idx_bssid]
            rssi = dic_['rssi'][idx_time, idx_bssid]
            resp = dic_['resp'][idx_time, idx_bssid]
            time = dic_['time'][idx_time]
            lst = dic_['lst'][idx_time, idx_bssid]
            d_time = dic_['time'][idx_time] - y[0]
            d_lst = dic_['lst'][idx_time, idx_bssid] - y[0]
        
        mask_t = np.expand_dims(bssid!=0, -2) * 1
        mask_b = bssid==0
#         subsequent_mask = np.triu(np.ones((CFG.n_time, CFG.n_time)), k=1).astype('uint8') == 0  
       
        return bssid.astype(np.int64), rssi.astype(np.int64), resp.astype(np.float64), time.astype(np.float64), lst.astype(np.float64),     d_time.astype(np.float64), d_lst.astype(np.float64), y.astype(np.float64), mask_b, mask_t

    
    def __len__(self):
        return len(self.l_idx)
    

train_db = ILNDataset(l_idx_train)
valid_db = ILNDataset(l_idx_valid, test=True)

train_loader = DataLoader(train_db, batch_size=CFG.batch_size, num_workers=1, shuffle=True)
valid_loader = DataLoader(valid_db, batch_size=CFG.batch_size, num_workers=1, shuffle=False)


# In[19]:


l_idx_train[:3]


# In[23]:


train_db[0]


# In[26]:


import copy
from typing import Optional, Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module, MultiheadAttention, ModuleList, Dropout, Linear, Linear, LayerNorm
from torch.nn.init import xavier_uniform_

class TransformerEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

#         if self.norm is not None:
#             output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
#         src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
#         src = self.norm2(src)
        return src
    
    
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)    


# In[27]:


dim_bssid = 128
dim_rssi = 16
dim_time = 1
d_model = 128
max_bssid = 239312 # df_100.bssid.max() + 1
max_rssi = 110
# max_b_id = 7020

class Transformer(Module):
    def __init__(self, d_model: int = d_model, nhead: int = 8, num_encoder_layers: int = 4,
                 dim_feedforward: int = d_model*4, dropout: float = 0.0, activation: str = "relu"):
        super(Transformer, self).__init__()
                 
        self.emb_bssid = nn.Embedding(max_bssid, dim_bssid)
        self.emb_rssi = nn.Embedding(max_rssi, dim_rssi)
        
#         self.norm = nn.LayerNorm(dim_bssid+dim_rssi+dim_time*3)
        self.v = nn.Linear(dim_bssid+dim_rssi+5, d_model)     
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, nn.LayerNorm(d_model))
        self._reset_parameters()          

        self.w = nn.Linear(d_model, CFG.n_tar * 3)  
                 
    def forward(self, bssid, rssi, resp, time, lst, d_time, d_lst, y, mask_b, mask_t):
        bssid = self.emb_bssid(bssid).type(torch.float64)
        rssi = self.emb_rssi(rssi).type(torch.float64)
        resp, time, lst, d_time, d_lst = resp.unsqueeze(-1), time.unsqueeze(-1), lst.unsqueeze(-1), d_time.unsqueeze(-1), d_lst.unsqueeze(-1)
        x = torch.cat([bssid, rssi, resp, time, lst, d_time, d_lst], dim=-1)
        
#         x = self.norm(x)
        x = self.v(x)
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1)
        x = x[:, -1] 

        x = self.w(x)         
        return x
    
    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask    

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


# In[28]:


size_batch = 4
bssid = torch.zeros([size_batch, CFG.n_time*CFG.n_bssid], dtype=torch.int64)
rssi = torch.zeros([size_batch, CFG.n_time*CFG.n_bssid], dtype=torch.int64)
resp = torch.zeros([size_batch, CFG.n_time*CFG.n_bssid], dtype=torch.float64)
time = torch.zeros([size_batch, CFG.n_time*CFG.n_bssid], dtype=torch.float64)
lst = torch.zeros([size_batch, CFG.n_time*CFG.n_bssid], dtype=torch.float64)
d_time = torch.zeros([size_batch, CFG.n_time*CFG.n_bssid], dtype=torch.float64)
d_lst = torch.zeros([size_batch, CFG.n_time*CFG.n_bssid], dtype=torch.float64)
y = torch.zeros([size_batch, 4])

mask_b = torch.ones([size_batch, CFG.n_bssid]) == 0
mask_t = torch.ones([size_batch, 1, CFG.n_time])

torch.set_default_dtype(torch.float64)
Transformer()(bssid, rssi, resp, time, lst, d_time, d_lst, y, mask_b, mask_t).shape


# ### Train

# In[30]:


import torch

def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        # works for nn.ConvNd and nn.Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
#         return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)
        return x.norm(norm_type, dim=tuple(range(0, x.ndim)), keepdim=True)


def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)


# In[31]:


def iln_loss(pred, y):
    return torch.mean(torch.abs(pred[:, 0] - y[:, 1]) * 15 + ((pred[:, 1] - y[:, 2]) ** 2 + (pred[:, 2] - y[:, 3]) ** 2) ** 0.5)

def iln_loss_valid(pred, y_int, y_t):
    pred, y_int, y_t = pred.detach().cpu().numpy(), y_int.detach().cpu().numpy(), y_t.detach().cpu().numpy()
    ar = np.zeros((len(y_t), 3))
    for i in range(len(y_t)):
        ar[i, 0] = np.interp(y_t[i, 0], y_int[i, :, 0], pred[i, :, 0])
        ar[i, 1] = np.interp(y_t[i, 0], y_int[i, :, 0], pred[i, :, 1])
        ar[i, 2] = np.interp(y_t[i, 0], y_int[i, :, 0], pred[i, :, 2])
        
    return np.mean(np.abs(ar[:, 0] - y_t[:, 1]) * 15 + ((ar[:, 1] - y_t[:, 2]) ** 2 + (ar[:, 2] - y_t[:, 3]) ** 2) ** 0.5)

# def iln_loss(pred, y):
#     return torch.mean(torch.abs(pred[:, 0] - y[:, 1]) * 15 + ((pred[:, 1] - y[:, 2]) ** 2 + (pred[:, 2] - y[:, 3]) ** 2) ** 0.5)


# In[32]:


def run_epoch(dataloaders, model, is_train):
    start = time.time()
    l_train_loss, l_valid_loss, l_tar, l_pred = [], [], [], []
    
    for i, (bssid, rssi, resp, t, lst, d_t, d_lst, y, mask_b, mask_t) in enumerate(dataloaders):
        bssid, rssi, resp, t, lst, d_t, d_lst, y, mask_b, mask_t         = bssid.cuda(), rssi.cuda(), resp.cuda(), t.cuda(), lst.cuda(), d_t.cuda(), d_lst.cuda(), y.cuda(), mask_b.cuda(), mask_t.cuda()
        
        with torch.set_grad_enabled(is_train):
            pred = model(bssid, rssi, resp, t, lst, d_t, d_lst, y, mask_b, mask_t)
            loss = iln_loss(pred, y)
            l_train_loss.append(loss.detach().cpu().numpy())
            
            if is_train:
                loss.backward()
                adaptive_clip_grad(model.parameters())
                opt.step()
                opt.zero_grad()
                if i % 1000 == 0:
                    print('Step: %d Loss: %.4f Time: %.0f' %(i, np.array(l_train_loss).mean(), time.time()-start))
                    start = time.time()                
            else:
                loss = iln_loss(pred, y)
                l_valid_loss.append(loss.detach().cpu().numpy())

    train_loss = np.array(l_train_loss).mean()
    
    if is_train:
        return train_loss
    else:
        valid_loss = np.array(l_valid_loss).mean()
        return train_loss, valid_loss


# In[34]:


import time
hist_loss_train, hist_loss_valid = {}, {}

def run():
    best_loss, best_epoch = 100, 0
    for epoch in range(1000):
        t = time.time()
        _ = model.train()
        train_loss = run_epoch(train_loader, model, True)
        hist_loss_train[epoch] = train_loss
        print('Epoch: %d Loss: %.4f Time: %0.f' %(epoch, train_loss, time.time()-t))
        
        t = time.time()
        _ = model.eval()        
        train_loss, valid_loss = run_epoch(valid_loader, model, False)
        hist_loss_valid[epoch] = valid_loss
        
        if valid_loss < best_loss:
#             torch.save(model.state_dict(), checkpoint_path)
            best_epoch = epoch
            best_loss = valid_loss
        print('Epoch: %d V_Loss: %.4f Best: %.4f %d' %(epoch, valid_loss, best_loss, best_epoch))
#         print('Epoch: %d V_Loss: %.4f Best: %.4f %d lr: %.6f' %(epoch, valid_loss, best_loss, best_epoch, scheduler.get_last_lr()[0]))
        print('')        
        
#         scheduler.step(valid_loss)
#         scheduler.step()
        
    return best_epoch, best_loss


# In[35]:


model = Transformer()
opt = torch.optim.AdamW(model.parameters(), lr=0.0003) 
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.9)
model.cuda()


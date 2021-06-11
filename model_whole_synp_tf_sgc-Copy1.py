#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import matplotlib.pyplot as plt

import json
import os
from sklearn.utils import shuffle

import pandas as pd

import random
import re
from nltk import sent_tokenize
from tqdm import tqdm
# import albumentations
# from albumentations.core.transforms_interface import DualTransform, BasicTransform

from IPython.display import display, HTML
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
display(HTML("<style>.container { width:90% !important; }</style>"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import bisect as bs
import matplotlib.pyplot as plt


# In[2]:


from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForPreTraining, BertConfig


# In[3]:


import nltk


# In[4]:


import os
gpu = '5'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu


# In[5]:


# from transformers import BertTokenizer, BertModel
# import torch

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state


# In[6]:


import pickle

#df.to_pickle('df_all.pkl')
df = pd.read_pickle('df_all.pkl')

# with open('conts_meta.pkl', 'wb') as f:
#     pickle.dump(conts_meta, f)

conts_meta = dict()
with open('conts_meta2.pkl', 'rb') as f:
    conts_meta = pickle.load(f)


# In[7]:


df = df.reset_index(drop = True)
df = df[df.cate.isin(['otv_movie', 'otv_drama'])]
df_asset_filter = pd.read_pickle('./data/df_movie_otv2.pkl')


# In[8]:


df = df.loc[((df.cate == 'otv_movie') & (df.conts_id.isin(df_asset_filter.mstr_conts_id.unique())))|(df.cate == 'otv_drama')]


# In[9]:


df = df.reset_index(drop = True)
df.conts_id.nunique()


# In[10]:


df_meta = pd.DataFrame.from_dict(conts_meta)
df_meta = df_meta.unstack().unstack()
df_meta = df_meta[df_meta.ply_sec != 0]
df_meta = df_meta[df_meta.ply_sec > 600]

df_meta['conts_id'] = df_meta.index
df_meta = df_meta.reset_index(drop = True)
df_meta.loc[df_meta.sesn_no == '_', 'sesn_no'] = 1
df_meta.loc[df_meta.epsd_tms == '_', 'epsd_tms'] = 1


# In[11]:


df_use = df.groupby(['conts_id', 'cate']).use_time.sum().reset_index()
df_buy = df.groupby(['conts_id', 'cate']).buy_amt.sum().reset_index()
df = df_use.merge(df_buy, on = ['conts_id', 'cate'])

df = df.merge(df_meta[['conts_id', 'asset_show_nm', 'ply_sec']], on = 'conts_id')
df = df[df.asset_show_nm != '']


# In[12]:


df['use_time'] = df.use_time.astype(int)
df['ply_sec'] = df.ply_sec.astype(int)
df['use_time'] = df.use_time/df.ply_sec

duplicated_asset_nm = df.loc[df[['conts_id', 'asset_show_nm']].groupby('asset_show_nm').conts_id.transform('count') > 1, ['asset_show_nm']].asset_show_nm.values
df1 = df[~df.asset_show_nm.isin(duplicated_asset_nm)]
df2 = df[df.asset_show_nm.isin(duplicated_asset_nm)]
df2['use_time'] = df2.groupby(['asset_show_nm', 'cate']).use_time.transform('sum')
df2['buy_amt'] = df2.groupby(['asset_show_nm', 'cate']).buy_amt.transform('sum')
df2 = df2.drop_duplicates(subset=['asset_show_nm'], keep='first')
df = pd.concat([df1, df2]).reset_index(drop=True)
df = df.drop('asset_show_nm', axis = 1).drop('ply_sec', axis = 1).merge(df_meta, on = 'conts_id')

# 01XXXX 실사영화
# 03XXXX tv시리즈
df = df[df.main_genre_cd.str.startswith('01') | df.main_genre_cd.str.startswith('03')]
df = df[~df.asset_show_nm.str.contains('캐치온쇼')]


# In[13]:


len(set(duplicated_asset_nm))


# In[14]:


df.use_time.quantile(np.arange(0, 1, 0.1)).plot()


# In[15]:


df.use_time.describe()


# In[16]:


l_quantile = df.use_time.quantile(np.arange(0, 1, 0.1)).values


# In[17]:


l_quantile


# In[18]:


l_quantile = df.use_time.quantile(np.arange(0, 1, 0.1)).values
df['use_grade'] = df.use_time.apply(lambda x : bs.bisect(l_quantile, x) - 1)


# In[19]:


df.head()


# In[20]:


#df['use_new_grade'] = (df.use_time - df.use_time.min()) / (df.use_time.max() - df.use_time.min())


# In[21]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# tfidf = TfidfVectorizer()
# tfidf_matrix = tfidf.fit_transform(df_meta.kywrd_str)
# tfidf_matrix.shape


# In[22]:


dic_cate, dic_cate_ = {}, {}
for i, j in enumerate(df.cate.unique()):
    dic_cate[i+1] = j
    dic_cate_[j] = i+1

df['cate_id'] = [dic_cate_[i] for i in df.cate.values]

dic_mgenre, dic_mgenre_ = {}, {}
for i, j in enumerate(df.main_genre_cd.unique()):
    dic_mgenre[i+1] = j
    dic_mgenre_[j] = i+1

df['mgenre_id'] = [dic_mgenre_[i] for i in df.main_genre_cd.values]

dic_sgenre, dic_sgenre_ = {}, {}
for i, j in enumerate(df.show_genre_cd.unique()):
    dic_sgenre[i+1] = j
    dic_sgenre_[j] = i+1

df['sgenre_id'] = [dic_sgenre_[i] for i in df.show_genre_cd.values]

dic_mnfc_yy, dic_mnfc_yy_ = {}, {}
for i, j in enumerate(df.mnfc_yy.unique()):
    dic_mnfc_yy[i+1] = j
    dic_mnfc_yy_[j] = i+1

df['mnfc_yy_id'] = [dic_mnfc_yy_[i] for i in df.mnfc_yy.values]


# In[23]:


df.kywrd_nm.head()


# In[24]:


df['kywrd_str'] = df.kywrd_nm.apply(lambda x : ' '.join(list(x)))


# In[25]:


p = re.compile('^\\[.+\\]\s|,_|_')
df['synps_mid_sbst'] = df.synps_mid_sbst.apply(lambda x : p.sub('', x) if p.match(x) else x)
df['synps_whole_sbst'] = df.synps_whole_sbst.apply(lambda x : p.sub('', x) if p.match(x) else x)


# In[26]:


(df['synps_whole_sbst'] == '').sum(), (df['synps_mid_sbst'] == '').sum()


# In[27]:


df.iloc[8]['synps_whole_sbst']


# In[28]:


df.iloc[8]['synps_mid_sbst']


# In[29]:


df['cine21_synp'] = df.cine21_synp.apply(lambda x : re.compile('\\r\\n|<b>|</b>|\(.{2,3}\)|‘|’|\u200b|,_|_').sub('', x))


# In[30]:


df['kofic_synp'] = df.kofic_synp.apply(lambda x : re.compile('\\r\\n|<b>|</b>|\(.{2,3}\)|‘|’|\u200b|,_|_').sub('', x))


# In[31]:


df['synps_sbst'] = df.apply(lambda x : x.synps_mid_sbst if x.synps_whole_sbst == '_' else x.synps_whole_sbst, axis = 1)


# In[32]:


(df['cine21_synp'].str.len() > 5).sum(), (df['kofic_synp'].str.len() > 5).sum()


# In[33]:


df[((df['cine21_synp'].str.len() > 0) & (df['kofic_synp'].str.len() == 0))]


# In[34]:


((df['cine21_synp'].str.len() > 5) & (df['cine21_synp'] == df['kofic_synp'])).sum()


# In[35]:


df['text'] = df.apply(lambda x : x.kywrd_str+x.synps_sbst+' '+ x.cine21_synp + ' ' + x.kofic_synp  if x.kofic_synp != x.cine21_synp else x.synps_sbst + x.kofic_synp, axis = 1)


# In[36]:


df.text.head()


# In[37]:


dic_actor, dic_actor_ = {}, {}
len_actor = 0
len_actor_seq = 0
for i, actors in enumerate(df.actr_nm.values):
    
    for j, actor in enumerate(list(actors)):
    
        len_dic = len(dic_actor)
    
        if j + 1 > len_actor_seq:
            len_actor_seq = j + 1
            
        if (actor in dic_actor_) == False:
            dic_actor[len_dic + 1] = actor
            dic_actor_[actor] = len_dic+1
    
        if len_actor < len(dic_actor):
            len_actor = len(dic_actor)
        
df['actor_id'] = [[dic_actor_[actor] for actor in actors] for i, actors in enumerate(df.actr_nm.values) ]
        
    
dic_writer, dic_writer_ = {}, {}
len_writer = 0
len_writer_seq = 0
for i, writers in enumerate(df.writer_nm.values):
    
    for j, writer in enumerate(list(writers)):
    
        len_dic = len(dic_writer)
        
        if j + 1 > len_writer_seq:
            len_writer_seq = j + 1

        if (writer in dic_writer_) == False:
            dic_writer[len_dic + 1] = actor
            dic_writer_[writer] = len_dic+1
    
        if len_writer < len(dic_writer):
            len_writer = len(dic_writer)
        
df['writer_id'] = [[dic_writer_[writer] for writer in writers] for i, writers in enumerate(df.writer_nm.values) ]

dic_dirt, dic_dirt_ = {}, {}
len_dirt = 0
len_dirt_seq = 0
for i, dirts in enumerate(df.dirt_nm.values):
    
    for j, dirt in enumerate(list(dirts)):
    
        len_dic = len(dic_dirt)
    
        if j + 1 > len_dirt_seq:
            len_dirt_seq = j + 1
            
        if (dirt in dic_dirt_) == False:
            dic_dirt[len_dic + 1] = actor
            dic_dirt_[dirt] = len_dic+1
    
        if len_dirt < len(dic_dirt):
            len_dirt = len(dic_dirt)
        
df['dirt_id'] = [[dic_dirt_[dirt] for dirt in dirts] for i, dirts in enumerate(df.dirt_nm.values) ]


# In[38]:


len_actor_seq, len_writer_seq, len_dirt_seq


# In[39]:


len_actor, len_writer, len_dirt


# In[40]:


dic_actor_conts = {}

for i, row in df.iterrows():

    for actor in list(row.actr_nm):

        if actor in dic_actor_conts:
            dic_actor_conts[actor] = dic_actor_conts[actor] + 1
        else:
            dic_actor_conts[actor] = 1
            
df_actor = pd.DataFrame.from_dict(dic_actor_conts, orient='index')
df_actor = df_actor.reset_index()
df_actor.columns = ['person', 'cnt']
df_actor = df_actor[df_actor.person != '_']


# In[41]:


df_actor


# In[42]:


df_actor.sort_values('cnt', ascending = False).head(20)


# In[ ]:





# #### label 설정

# In[43]:


l_quantile = df.buy_amt.quantile(np.arange(0, 1, 0.1)).values.astype('int'); l_quantile
# l_quantile = np.linspace(l_quantile[0], l_quantile[-2], 10).astype('int'); l_quantile
df['label_buy_amt'] = df.buy_amt.apply(lambda x : bs.bisect(l_quantile, x) - 1)
df.label_buy_amt.value_counts().sort_index().values


# In[45]:


l_quantile = df.use_time.quantile(np.arange(0, 1, 0.1)).values.astype('int'); l_quantile
l_quantile = np.linspace(l_quantile[0], l_quantile[-1], 10).astype('int'); l_quantile
df['label_use_time'] = df.use_time.apply(lambda x : bs.bisect(l_quantile, x) - 1)
df.label_use_time.value_counts().sort_index().values


# In[398]:


# l_quantile = df.use_time.quantile(np.sin(np.linspace(0, np.pi/2, 11)[:-1])).values.astype('int'); l_quantile
# df['label_use_time'] = df.use_time.apply(lambda x : bs.bisect(l_quantile, x) - 1)
# df.label_use_time.value_counts().sort_index().values


# In[232]:


l_quantile = df[df.cate=='otv_movie'].use_time.quantile(np.arange(0, 1, 0.1)).values.astype('int'); l_quantile
l_quantile = np.linspace(l_quantile[0], l_quantile[-1], 10).astype('int'); l_quantile
df.loc[df.cate=='otv_movie', 'label_use_time'] = df.loc[df.cate=='otv_movie', 'use_time'].apply(lambda x : bs.bisect(l_quantile, x) - 1)
df.loc[df.cate=='otv_movie', 'label_use_time'].value_counts().sort_index().values


# In[233]:


l_quantile = df[df.cate=='otv_drama'].use_time.quantile(np.arange(0, 1, 0.1)).values.astype('int'); l_quantile
l_quantile = np.linspace(l_quantile[0], l_quantile[-1], 10).astype('int'); l_quantile
df.loc[df.cate=='otv_drama', 'label_use_time'] = df.loc[df.cate=='otv_drama', 'use_time'].apply(lambda x : bs.bisect(l_quantile, x) - 1)
df.loc[df.cate=='otv_drama', 'label_use_time'].value_counts().sort_index().values


# In[ ]:





# #### feature 추출

# In[419]:


all_db = ContentDataset(df, df_meta, False)#, transforms=transforms())
all_loader = DataLoader(all_db, batch_size=batch_size, shuffle=False,  num_workers=num_workers, pin_memory=True)

model.load_state_dict(torch.load('w_sgc.pth'))  

l_feature = []
for i, (inp_ids, inp_mask, inp2, tar_cls, tar) in enumerate(all_loader): 
    inp_ids, inp_mask, inp2, tar = inp_ids.to(device), inp_mask.to(device), inp2.to(device).long(), tar.to(device)

    inp_ids = inp_ids.squeeze(1)
    inp_mask = inp_mask.squeeze(1)
    inp2 = inp2.squeeze(1)

    with torch.no_grad():
        pred, feature = model.forward(inp_ids, inp_mask, inp2)

    l_feature.append(feature.detach().cpu().numpy())
    
feature = np.concatenate(l_feature, axis=0)


# In[202]:


feature.shape


# ### LGBM +

# In[46]:


df_ = df.loc[:, ['conts_id', 'cate', 'use_time', 'buy_amt', 
       'first_frmtn_date',
       'dirt_nm', 'actr_nm', 'writer_nm',
       'main_genre_cd', 'show_genre_cd',
       'mnfc_yy', 'ply_sec', 'sesn_no', 'epsd_tms',
       'use_grade', 'label_use_time', 'label_buy_amt']]


# In[47]:


df_['ff_yy'] = df_.first_frmtn_date.apply(lambda x: int(x[:4]))


# In[48]:


def make_id(d_, col_ori, col):
    dic_ = {}
    for i, k in enumerate(d_[col_ori].unique()):
        dic_[k] = i+1
    d_[col] = d_[col_ori].apply(lambda x: dic_[x])   
    return dic_

def make_id_from_set(d_, col_ori, col):
    dic_ = {}
    
# 모든 사람에 대해서 dic_ 구성
#     for i, k in enumerate(set().union(*d_[col_ori])):
#         dic_[k] = i+1
        
# x번 이상 나오는 사람에 대해서 dic_ 구성
    l_p = [p for l in d_[col_ori] for p in l]
    l_p = [p for p in set(l_p) if l_p.count(p) >= 2]      
    for i, k in enumerate(l_p):
        dic_[k] = i+1       

    d_[col] = d_[col_ori].apply(lambda x: [dic_[p] for p in x if p in dic_.keys()]) 

    return dic_

dic_cate = make_id(df_, 'cate', 'cate_id')
dic_ff = make_id(df_, 'ff_yy', 'ff_yy_id')
dic_actor = make_id_from_set(df_, 'actr_nm', 'actor_id')
dic_writer = make_id_from_set(df_, 'writer_nm', 'writer_id')
dic_dirt = make_id_from_set(df_, 'dirt_nm', 'dirt_id')
dic_main_gen = make_id(df_, 'main_genre_cd', 'main_gen_id')
dic_show_gen = make_id(df_, 'show_genre_cd', 'show_gen_id')
dic_yy = make_id(df_, 'mnfc_yy', 'yy_id')
dic_sesn = make_id(df_, 'sesn_no', 'sesn_id')
dic_epsd = make_id(df_, 'epsd_tms', 'epsd_id')    


# In[49]:


df_.iloc[:2, -15:]


# In[50]:


def make_one_hot(d_, dic, col):
    x = np.zeros((len(d_), len(dic)+1))
    for i, l in enumerate(d_[col]):
        if isinstance(l, int):
            x[i, l] = 1  
        elif len(l) > 0:
            x[i, l] = 1  
    return x

l_x = []
l_x.append(make_one_hot(df_, dic_cate, 'cate_id'))
l_x.append(make_one_hot(df_, dic_ff, 'ff_yy_id'))
l_x.append(make_one_hot(df_, dic_actor, 'actor_id'))
l_x.append(make_one_hot(df_, dic_writer, 'writer_id'))
l_x.append(make_one_hot(df_, dic_dirt, 'dirt_id'))
l_x.append(make_one_hot(df_, dic_main_gen, 'main_gen_id'))
l_x.append(make_one_hot(df_, dic_show_gen, 'show_gen_id'))
l_x.append(make_one_hot(df_, dic_yy, 'yy_id'))
l_x.append(make_one_hot(df_, dic_sesn, 'sesn_id'))
l_x.append(make_one_hot(df_, dic_epsd, 'epsd_id'))
x = np.concatenate(l_x, axis=-1)


# In[51]:


len(dic_cate), len(dic_ff), len(dic_actor), len(dic_writer), len(dic_dirt), len(dic_main_gen), len(dic_show_gen), len(dic_yy), len(dic_sesn), len(dic_epsd)


# In[52]:


x = np.concatenate([x, df_.loc[:, ['ply_sec']].values], axis=-1)
y = df_.loc[:, 'label_use_time'].values
# y = df_.loc[:, 'label_buy_amt'].values
x.shape


# In[53]:


# x = np.concatenate([x, feature], axis=-1)
# x.shape


# In[54]:


from sklearn.model_selection import train_test_split
conts_id_list = df.conts_id.unique()
len(conts_id_list)

train_conts_ids, valid_conts_ids = train_test_split(conts_id_list, test_size=0.3, random_state=1, stratify = df.use_grade)
len(train_conts_ids), len(valid_conts_ids)


# In[55]:


x_train, y_train = x[df_.conts_id.isin(train_conts_ids)], y[df_.conts_id.isin(train_conts_ids)]
x_valid, y_valid = x[df_.conts_id.isin(valid_conts_ids)], y[df_.conts_id.isin(valid_conts_ids)]


# In[56]:


x_train.shape, x_valid.shape


# In[57]:


import lightgbm as lgb

train_data = lgb.Dataset(x_train, label=y_train)
valid_data = lgb.Dataset(x_valid, label=y_valid)

param = {'objective': 'regression_l1', 'learning_rate': 0.1, 'num_leaves':31}
param['metric'] = 'l1'


# In[58]:


bst = lgb.train(param, train_data, 10000, [train_data, valid_data], early_stopping_rounds=3000)


# In[255]:


bst = lgb.train(param, train_data, 10000, [train_data, valid_data], early_stopping_rounds=3000)


# In[247]:


bst = lgb.train(param, train_data, 10000, [train_data, valid_data], early_stopping_rounds=3000)


# In[195]:


bst = lgb.train(param, train_data, 10000, [train_data, valid_data], early_stopping_rounds=3000)


# In[59]:


pred_lgbm = bst.predict(x_valid, num_iteration=bst.best_iteration)


# In[60]:


pd.Series(bst.feature_importance()).sort_values(ascending=False).iloc[:50]


# In[61]:


# 정확도 계산 함수
def flat_accuracy(preds, labels):
    
    pred_flat = preds.flatten().round()
    labels_flat = labels.flatten().astype(int)
    
    acc = np.count_nonzero(pred_flat - labels_flat == 0) / len(labels_flat)
    acc1 = np.count_nonzero(np.abs(pred_flat - labels_flat) == 1) / len(labels_flat)
    acc2 = np.count_nonzero(np.abs(pred_flat - labels_flat) == 2) / len(labels_flat)
    
    return acc, acc1, acc2


# In[62]:


flat_accuracy(pred_lgbm, y_valid) # l1loss


# In[252]:


flat_accuracy(pred_lgbm, y_valid) # l2loss


# In[193]:


flat_accuracy(pred_lgbm_l1, y_valid) # l1loss


# In[198]:


flat_accuracy(pred_lgbm, y_valid) # l2loss


# In[200]:


flat_accuracy((pred_lgbm + pred_lgbm_l1)/2, y_valid)


# In[63]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.clip(pred_lgbm, 0, 9).round(0), y_valid)
cm


# In[436]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.clip(pred_lgbm, 0, 9).round(0), y_valid)
cm


# In[437]:


import seaborn as sn

plt.figure(figsize = (10,8))
sn.heatmap(cm, annot=True)


# ### LGBM-

# In[ ]:





# In[ ]:





# In[ ]:





# ### Transformer+

# In[64]:


# 정확도 계산 함수
def flat_accuracy(preds, labels):
    
    pred_flat = preds.flatten().round()
    labels_flat = labels.flatten().astype(int)
    
    acc = np.count_nonzero(pred_flat - labels_flat == 0) / len(labels_flat)
    acc1 = np.count_nonzero(np.abs(pred_flat - labels_flat) == 1) / len(labels_flat)
    acc2 = np.count_nonzero(np.abs(pred_flat - labels_flat) == 2) / len(labels_flat)
    
    return acc, acc1, acc2


# In[65]:


np.random.seed(1)

class CFG:
    seq_len = 20
    batch_size=64
    n_feature = 3504


# In[66]:


x_train.shape, x_valid.shape


# In[67]:


x_all = np.concatenate([x_train, x_valid])
y_all = np.concatenate([y_train, y_valid])
x_all.shape, y_all.shape


# In[68]:


ptr_idx_train = []
for i in range(len(x_train) - CFG.seq_len + 1):
    ptr_idx_train.append((i, i+CFG.seq_len))

def get_l_idx_train(idx_train):
    l_idx_train = []
    for i in range(len(ptr_idx_train)):
        s, e = ptr_idx_train[i][0], ptr_idx_train[i][1]
        l_idx_train.append(idx_train[s:e])    
    return np.array(l_idx_train)


l_idx_train = get_l_idx_train(np.arange(len(x_train)))

ptr_idx_valid = np.random.choice(len(l_idx_train), len(x_valid), replace=False)
l_idx_valid = l_idx_train[ptr_idx_valid]
l_idx_valid[:, -1] = np.arange(len(x_train), len(x_all))


# In[69]:


l_idx_train


# In[70]:


l_idx_valid


# In[71]:


l_idx_train.shape, l_idx_valid.shape


# In[72]:


class OCDataset(Dataset):
    def __init__(self, l_idx, test=False):
        self.l_idx = l_idx
        self.test = test
        
    def __getitem__(self, idx):
        idx = self.l_idx[idx] 
        
        x = x_all[idx]
        x1 = np.zeros((CFG.seq_len, 1))
        x1[1:, 0] = y_all[idx][:-1]
        x = np.concatenate([x, x1], axis=-1)
        
        y = y_all[idx].reshape(-1, 1)

        return x.astype(np.float32), y.astype(np.float32)

    
    def __len__(self):
        return len(self.l_idx)
    

train_db = OCDataset(l_idx_train)
valid_db = OCDataset(l_idx_valid, test=True)

train_loader = DataLoader(train_db, batch_size=CFG.batch_size, num_workers=1, shuffle=True)
valid_loader = DataLoader(valid_db, batch_size=CFG.batch_size, num_workers=1, shuffle=False)


# In[73]:


train_db[0]


# In[74]:


train_db[0][0].shape, train_db[0][1].shape


# In[75]:


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

        if self.norm is not None:
            output = self.norm(output)

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
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
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


# In[76]:


d_model = 512

class Transformer(Module):
    def __init__(self, d_model: int = d_model, nhead: int = 8, num_encoder_layers: int = 4,
                 dim_feedforward: int = d_model*4, dropout: float = 0.0, activation: str = "relu"):
        super(Transformer, self).__init__()
                 
#         self.norm = nn.LayerNorm(CFG.n_feature)
        self.v = nn.Linear(CFG.n_feature, d_model)     
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, nn.LayerNorm(d_model))
        self._reset_parameters()          

        self.w = nn.Linear(d_model, 1)  
                 
    def forward(self, x, mask):
#         x = self.norm(x)
        x = self.v(x)
        x = x.transpose(0, 1)
        x = self.encoder(x, mask)
        x = x.transpose(0, 1)
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


# In[77]:


def generate_square_subsequent_mask(sz: int) -> Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask 

generate_square_subsequent_mask(10)


# In[78]:


size_batch = 4
x = torch.zeros([size_batch, CFG.seq_len, CFG.n_feature], dtype=torch.float)
mask = generate_square_subsequent_mask(CFG.seq_len)
# torch.set_default_dtype(torch.float64)
Transformer()(x, mask).shape


# In[85]:


def run_epoch(dataloaders, model, is_train):
    start = time.time()
    l_train_loss, l_valid_loss, l_tar = [], [], []
    ar_pred, ar_y = np.empty((0, 1)), np.empty((0, 1))
    mask = model.generate_square_subsequent_mask(CFG.seq_len)
    
    for i, (x, y) in enumerate(dataloaders):
        x, y, mask = x.cuda(), y.cuda(), mask.cuda()
        
        with torch.set_grad_enabled(is_train):
            pred = model(x, mask)
            loss = nn.MSELoss()(pred, y)
            l_train_loss.append(loss.detach().cpu().numpy())
            
            if is_train:
                loss.backward()
#                 adaptive_clip_grad(model.parameters())
                opt.step()
                opt.zero_grad()
                if i % 10 == 0:
                    print('Step: %d Loss: %.4f Time: %.0f' %(i, np.array(l_train_loss).mean(), time.time()-start))
                    start = time.time()                
            else:
                pred, y = pred[:, -1], y[:, -1]
                loss = nn.MSELoss()(pred, y)
                l_valid_loss.append(loss.detach().cpu().numpy())
                ar_pred = np.concatenate([ar_pred, pred.detach().cpu().numpy()])
                ar_y= np.concatenate([ar_y, y.detach().cpu().numpy()])

    train_loss = np.array(l_train_loss).mean()
    
    if is_train:
        return train_loss
    else:
        valid_loss = np.array(l_valid_loss).mean()
        return train_loss, valid_loss, ar_pred, ar_y


# In[86]:


import time
hist_loss_train, hist_loss_valid = {}, {}

def run():
    global l_pred, l_y
    best_loss, best_epoch = 100, 0
    for epoch in range(1000):
        t = time.time()
        _ = model.train()
        
        ar = np.arange(len(x_train))
        if epoch > 0:
            np.random.shuffle(ar)
            
        l_idx_train = get_l_idx_train(ar)
        train_db = OCDataset(l_idx_train)
        train_loader = DataLoader(train_db, batch_size=CFG.batch_size, num_workers=1, shuffle=True)
        
        train_loss = run_epoch(train_loader, model, True)
        hist_loss_train[epoch] = train_loss
        print('Epoch: %d Loss: %.4f Time: %0.f' %(epoch, train_loss, time.time()-t))
        
        t = time.time()
        _ = model.eval()        
        train_loss, valid_loss, l_pred, l_y = run_epoch(valid_loader, model, False)
        hist_loss_valid[epoch] = valid_loss
        
        if valid_loss < best_loss:
#             torch.save(model.state_dict(), checkpoint_path)
            best_epoch = epoch
            best_loss = valid_loss
#         print('Epoch: %d V_Loss: %.4f Best: %.4f %d' %(epoch, valid_loss, best_loss, best_epoch))
        print('Epoch: %d V_Loss: %.4f Best: %.4f %d lr: %.6f' %(epoch, valid_loss, best_loss, best_epoch, scheduler.get_last_lr()[0]))
        print(flat_accuracy(l_pred, l_y))    
        print('')        
        
#         scheduler.step(valid_loss)
        scheduler.step()
        
    return best_epoch, best_loss


# In[87]:


model = Transformer()
opt = torch.optim.AdamW(model.parameters(), lr=0.0001) 
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)
model.cuda()


# In[ ]:


run() # seq 20 512-4 (1, 10, 0.97) l2loss linspace


# In[84]:


run() # seq 20 512-4 (1, 10, 0.97) l1loss linspace


# In[82]:


run() # seq 20 512-4 (1, 10, 0.99) linspace


# In[219]:


run() # seq 50 512-4 (1, 10, 0.95)


# In[185]:


run() # seq 2 512-4 (1, 10, 0.99)


# In[91]:


run() # seq 10 512-8 (1, 10, 0.99)


# In[84]:


run() # seq 10 512 (2, 10, 0.95)


# In[82]:


run() # seq 10 512 (2, 10, 0.9)


# In[75]:


run() # seq 500 512 (2, 10, 0.9)


# In[ ]:


run() # seq 100 512 lr 0.0001


# In[80]:


run()


# In[ ]:


run()


# In[242]:


run()


# In[243]:


l_pred.shape, l_y.shape


# In[246]:


l_pred.squeeze()


# In[248]:


flat_accuracy(l_pred, l_y)


# ### Transformer-

# In[ ]:





# In[ ]:





# In[ ]:





# ### Syn+

# In[438]:


from sklearn.model_selection import train_test_split

conts_id_list = df.conts_id.unique()
len(conts_id_list)

train_conts_ids, valid_conts_ids = train_test_split(conts_id_list, test_size=0.3, random_state=1, stratify = df.use_grade)
len(train_conts_ids), len(valid_conts_ids)


# In[439]:


df_train = df[df.conts_id.isin(train_conts_ids)]
df_valid = df[df.conts_id.isin(valid_conts_ids)]


# In[440]:


df.head(1).text, df.head(1).synps_mid_sbst


# In[441]:


MAX_LENGTH = 512 #150 #100  #200
# model_path = '../../project/OContents/xlm-roberta-large'
model_path = '../../project/OContents/bert-base-multilingual-cased'
label = 'label_use_time'

class ContentDataset(Dataset):
    def __init__(self, df, df_meta, test=False):
        self.conts_list = df.conts_id.values
        self.df = df
        self.test = test
#         if test==False: self.label = df['use_time'].values/10e8  
        if test==False: self.label = df[label].values    
        
        if test==False: self.label_cls = df[label].values  
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)     
        
    def __getitem__(self, idx):  
        conts_id = self.conts_list[idx]
        #text = self.conts_meta.loc[self.conts_meta.conts_id == conts_id, 'synps_mid_sbst'].values[0]
#         text = self.df.loc[self.df.conts_id == conts_id, 'text'].values[0]
        text = self.df.loc[self.df.conts_id == conts_id, 'synps_mid_sbst'].values[0]
        
        #text, _ = self.transforms(data=(text, 'en'))['data']
        
        cate_id = self.df.loc[self.df.conts_id == conts_id, 'cate_id'].values[0]
        mgenre_id = self.df.loc[self.df.conts_id == conts_id, 'mgenre_id'].values[0]
        sgenre_id = self.df.loc[self.df.conts_id == conts_id, 'sgenre_id'].values[0]
        mnfc_yy_id = self.df.loc[self.df.conts_id == conts_id, 'mnfc_yy_id'].values[0]
        
        writer_id_list = self.df.loc[self.df.conts_id == conts_id, 'writer_id'].values[0]
        writer_id = np.zeros(len_writer_seq + 1)
        writer_id[0:len(writer_id_list)] = writer_id_list
        
        dirt_id_list = self.df.loc[self.df.conts_id == conts_id, 'dirt_id'].values[0]
        dirt_id = np.zeros(len_dirt_seq + 1)
        dirt_id[0:len(dirt_id_list)] = dirt_id_list
        
        actor_id_list = self.df.loc[self.df.conts_id == conts_id, 'actor_id'].values[0]
        actor_id = np.zeros(len_actor_seq + 1)
        actor_id[0:len(actor_id_list)] = actor_id_list

        
#         text = "[CLS] " + text + " [SEP]"
        
#         print(text)
#         if self.transforms:
#             text, _ = self.transforms(data=(self.conts_meta[self.conts_list[idx]]['synps_mid_sbst'], 'kr'))['data']   
#         print(text)
        encoded = self.tokenizer.encode_plus(text, 
                                             add_special_tokens=True,
                                             max_length=MAX_LENGTH,
                                             truncation=True,
                                             padding='max_length',
                                             return_tensors="pt")
#         print(len(text), (encoded['input_ids']!=0).sum(), int((encoded['input_ids']!=0).sum())/len(text))

        input2 = np.expand_dims(np.concatenate((np.array([cate_id, mgenre_id, sgenre_id, mnfc_yy_id]), writer_id, dirt_id, actor_id), axis = 0), 0)

        if self.test:
            return encoded['input_ids'], encoded['attention_mask'], input2
        
        
#             return encoded['input_ids'], encoded['attention_mask']
        else:
            return encoded['input_ids'], encoded['attention_mask'], input2, self.label_cls[idx], self.label[idx]
#             return encoded['input_ids'], encoded['attention_mask'], self.label[idx]
            

    def __len__(self):
        return len(self.conts_list)


# In[442]:


# tokenizer = AutoTokenizer.from_pretrained('../../project/OContents/xlm-roberta-large')  
# inputs = tokenizer("Hello, my dog is cute", max_length=512, return_tensors="pt")
# inputs['input_ids'], inputs['attention_mask']     
# model = AutoModel.from_pretrained(model_nm)
# output = model(**inputs)
# output['last_hidden_state'].shape
# output['pooler_output'].shape

# #[vocab_[input_id]  for input_id in inputs['input_ids'][0]]

# for input_id in inputs['input_ids'][0].numpy():    
#     vocab_[input_id]


# In[443]:


batch_size = 3 #32 #8 #16 #32
num_workers = 0

train_db = ContentDataset(df_train, df_meta, False)#, transforms=transforms())
valid_db = ContentDataset(df_valid, df_meta, False)

train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_db, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# In[444]:


train_db[0]


# In[445]:


for i, (inp_ids, inp_mask, inp2, tar_cls, tar) in enumerate(train_loader):
    
    break


# In[446]:


inp_ids.shape, inp_mask.shape, inp2.shape


# In[447]:


class ContentModel(nn.Module):  
    def __init__(self, backbone, dropout=0.0, num_class=1):
        super(ContentModel, self).__init__()        
        self.backbone = backbone
        self.v = nn.Linear(self.backbone.pooler.dense.out_features, 1)
        
    def forward(self, input_ids, attention_mask, x2):        
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)  #x: (batch, seq, emb_size) x2: (batch, emb_size)             
        x_last = output['last_hidden_state'] # (batch, seq_len, dim_bert)
        x_pool = output['pooler_output']  # (batch, dim_bert)    
        return self.v(x_pool)


# In[53]:


class ContentModel(nn.Module):  
    def __init__(self, backbone, dropout=0.0, num_class=1):
        super(ContentModel, self).__init__()        
        self.backbone = backbone
#         self.dropout = nn.Dropout(dropout)        
# #         self.w = nn.Sequential(nn.Linear(num_class+16, num_class))

        self.cate_emb = nn.Embedding(len(dic_cate_)+1, 2)
        self.mgenre_emb = nn.Embedding(len(dic_mgenre_)+1, 4)
        self.sgenre_emb = nn.Embedding(len(dic_sgenre_)+1, 4)
        self.mnfc_yy_emb = nn.Embedding(len(dic_mnfc_yy_)+1, 4)
        
        #(배치 크기, 시퀀스 길이) => (배치 크기, 시퀀스 길이, 임베딩 차원)
        
        self.writer_emb = nn.Embedding(len(dic_writer_)+1, 64)
        self.dirt_emb = nn.Embedding(len(dic_dirt_)+1, 64)
        self.actor_emb = nn.Embedding(len(dic_actor_)+1, 64)
        
#         self.w = nn.Linear(2*self.backbone.pooler.dense.out_features + (2+4*3+64*3), 512)

#         self.w1 = nn.Linear(512, 128)
#         self.w2 = nn.Linear(128, 64)
#         self.w3 = nn.Linear(64, 16)
# #         self.w4 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
#         self.w4 = nn.Sequential(nn.Linear(16, 1))
    
        self.v = nn.Linear(self.backbone.pooler.dense.out_features+206, 1)
    def forward(self, input_ids, attention_mask, x2):        
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)  #x: (batch, seq, emb_size) x2: (batch, emb_size)             
        x_last = output['last_hidden_state'] # (batch, seq_len, dim_bert)
        x_pool = output['pooler_output']  # (batch, dim_bert)    
#         return self.v(x_pool), x_pool

        cate = self.cate_emb(x2[:, 0])
        mgenre = self.mgenre_emb(x2[:, 1])
        sgenre = self.sgenre_emb(x2[:, 2])
        mnfc_yy = self.mnfc_yy_emb(x2[:, 3])
        
        #len_actor_seq, len_writer_seq, len_dirt_seq

        start_idx = 4
        end_idx = start_idx + len_writer_seq
        writer = self.writer_emb(x2[:, start_idx:end_idx])
        
        start_idx = end_idx
        end_idx = start_idx + len_dirt_seq
        dirt = self.dirt_emb(x2[:, start_idx:end_idx])
        
        start_idx = end_idx
        end_idx = start_idx + len_actor_seq
        actor = self.actor_emb(x2[:, start_idx:end_idx])
    
        x = torch.cat([cate, mgenre, sgenre, mnfc_yy, torch.mean(writer, 1), torch.mean(dirt, 1), torch.mean(actor, 1)], axis=-1)
        x = torch.cat([x, x_pool], axis=-1)
        return self.v(x)
#         x = torch.cat([x, x_pool, torch.mean(x_last, 1)], axis=-1)
#         x = self.dropout(x)
#         x = self.w(x)
#         x = self.dropout(x)
#         x = self.w1(x)
#         x = self.dropout(x)
#         x = self.w2(x)
#         x = self.dropout(x)
#         x = self.w3(x)
#         x = self.w4(x)
        
        
#         return x


# In[448]:


for i, (inp_ids, inp_mask, inp2, tar_cls, tar) in enumerate(train_loader):
    
    break
# inp_ids, inp_mask, inp2, tar = inp_ids.to(device), inp_mask.to(device), inp2.to(device).long(), tar.to(device)

inp_ids = inp_ids.squeeze(1)
inp_mask = inp_mask.squeeze(1)
inp2 = inp2.long().squeeze(1)


# In[449]:


(ContentModel(backbone=AutoModel.from_pretrained(model_path, num_labels = 10), dropout=0.3, num_class=10)(inp_ids, inp_mask, inp2)).shape


# In[450]:


from transformers import BertForSequenceClassification

config = BertConfig.from_json_file(model_path+'/config.json')
model = ContentModel(backbone=AutoModel.from_pretrained(model_path), dropout=0.)   

opt = torch.optim.AdamW(model.parameters(), lr=0.00001)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)  


# In[451]:


model


# In[452]:


for i, param in enumerate(model.backbone.parameters()):
    print(i, param.shape)
    if i < 181:
        param.requires_grad = False


# In[453]:


for i, param in enumerate(model.parameters()):
    print(i, param.requires_grad, param.shape)    


# In[454]:


# tokenizer = AutoTokenizer.from_pretrained('../../project/OContents/bert-base-multilingual-cased') 
# text = '1,600만의 선택! 지금까지 이런 흥행은 없었다! 낮에는 치킨장사, 밤에는 잠복근무. 해체 위기를 맞는 마약반 5인방은 감시를 위해 범죄조직의 아지트 앞 치킨집을 인수해 위장 창업을 하게 되고 치킨집은 맛집으로 입소문 나기 시작한다. 범인을 잡을 것인가, 닭을 잡을 것인가!'
# encoded = tokenizer.encode_plus(text, add_special_tokens=True,
#                                     max_length=MAX_LENGTH,
#                                     return_tensors="pt")


# In[61]:


#[tokenizer.convert_ids_to_tokens(s) for s in encoded['input_ids'].tolist()[0]]


# In[455]:


# 정확도 계산 함수
def flat_accuracy(preds, labels):
    
    pred_flat = preds.flatten().round()
    labels_flat = labels.flatten().astype(int)
    
    acc = np.count_nonzero(pred_flat - labels_flat == 0) / len(labels_flat)
    acc1 = np.count_nonzero(np.abs(pred_flat - labels_flat) == 1) / len(labels_flat)
    acc2 = np.count_nonzero(np.abs(pred_flat - labels_flat) == 2) / len(labels_flat)
    
    return acc, acc1, acc2


# In[456]:


# 정확도 계산 함수
def mse_flat_accuracy(preds, labels):
    
    pred_flat = robustScaler.inverse_transform(np.array(preds.flatten()).reshape(-1, 1))
    pred_flat = [bs.bisect(l_quantile, x) - 1 for x in pred_flat]
    
    labels_flat = labels.flatten().astype(int)
    
    acc = np.count_nonzero(pred_flat - labels_flat == 0) / len(labels_flat)
    acc1 = np.count_nonzero(np.abs(pred_flat - labels_flat) == 1) / len(labels_flat)
    acc2 = np.count_nonzero(np.abs(pred_flat - labels_flat) == 2) / len(labels_flat)
    
    return acc, acc1, acc2


# In[457]:


import torch

def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
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


# In[458]:


## loss label smoothing
def run_epoch(dataloaders, model, is_train):
    "Standard Training and Logging Function"
    start = time.time()
    best_acc = 0.0
    l_total_loss, l_total_acc = [], []
    l_pred, l_tar, l_tar_cls = [], [], []
        
    for i, (inp_ids, inp_mask, inp2, tar_cls, tar) in enumerate(dataloaders): 
        inp_ids, inp_mask, inp2, tar = inp_ids.to(device), inp_mask.to(device), inp2.to(device).long(), tar.to(device)
        
#         torch.tensor(b_input_ids).to(device).long()
        inp_ids = inp_ids.squeeze(1)
        inp_mask = inp_mask.squeeze(1)
        inp2 = inp2.squeeze(1)

        with torch.set_grad_enabled(is_train):
            pred = model.forward(inp_ids, inp_mask, inp2)
            pred = pred.squeeze(1)    
#             loss = torch.nn.L1Loss()(pred.float(), tar.float()) 
            loss = torch.nn.MSELoss()(pred.float(), tar.float()) 
            
#             pred = model.forward(inp_ids, inp_mask, inp2)
#             loss = nn.CrossEntropyLoss()(pred, tar.type(torch.long))
    
            l_total_loss.append(loss.detach().cpu().numpy())

            if is_train:
                loss.backward()
#                 adaptive_clip_grad(model.parameters())
                opt.step()
                opt.zero_grad()
                
                if i % 100 == 1:
                    elapsed = time.time() - start
                    print("Epoch Step: %d Loss: %f Time: %f" %   (i, np.array(l_total_loss).mean(), elapsed))
                    start = time.time()
            else:
                
                l_pred.append(pred.detach().cpu().numpy())
                l_tar.append(tar.detach().cpu().numpy())
                l_tar_cls.append(tar_cls.detach().cpu().numpy())
                
                
    total_loss = np.array(l_total_loss).mean()
    
    if is_train:
        return total_loss
    else:
        y_pred = np.concatenate(l_pred)
        y_test = np.concatenate(l_tar)        

        total_acc, total_acc1, total_acc2 =  flat_accuracy(y_pred, y_test)
        return total_loss, total_acc, total_acc1, total_acc2


# In[459]:


def run(model, train_loader, valid_loader, n_epoch=30, save_path='w_sgc.pth'):  
    best = {"loss": 1e10, "acc" : 0.0, "epoch" : 0}
    

    for epoch in range(n_epoch):
        t = time.time()
        _ = model.train()
        total_loss = run_epoch(train_loader, model, True)
        print("Epoch: %d  Loss: %f  Time: %f" % (epoch, total_loss, time.time()-t))
        _ = model.eval()
        total_loss, total_acc, total_acc1, total_acc2 = run_epoch(valid_loader, model, False)

        if total_acc > best["acc"]:    
            best["acc"] = total_acc
            best["epoch"] = epoch
            torch.save(model.state_dict(), save_path)
            
        scheduler.step()

        print("Epoch: %d  Loss: %f  Acc: %.4f Acc1: %.4f Acc2: %.4f Best Epoch: %d Best Acc: %.4f" % (epoch, total_loss, total_acc, total_acc1, total_acc2, best["epoch"], 
                                                                                best["acc"]))
        print('')


# In[460]:


# label_use_time train 181~ lr 0.00001-10-0.9 # cate 제거
run(model, train_loader, valid_loader, 500) 


# In[67]:


# label_use_time train 181~ lr 0.00001-10-0.9 # cate 추가
run(model, train_loader, valid_loader, 500) 


# In[179]:


# label_use_time train 181~ lr 0.00001-10-0.9 # feature extraction 추가
run(model, train_loader, valid_loader, 500) 


# In[131]:


# label_use_time train 181~ lr 0.00001-10-0.9
run(model, train_loader, valid_loader, 500) 


# In[76]:


# use_grade train 181~ lr 0.00001-10-0.9
run(model, train_loader, valid_loader, 500) 


# In[ ]:


# train 0~ lr 0.00001-10-0.9 agc 0.01
run(model, train_loader, valid_loader, 500) 


# In[ ]:


# lr 0.00001-10-0.9 Best Epoch: 65 Best Acc: 0.2211
run(model, train_loader, valid_loader, 500) 


# In[148]:


# lr 0.00001 - 아래에서 계속 2065
run(model, train_loader, valid_loader, 50) 


# In[94]:


# 
run(model, train_loader, valid_loader, 50) 


# In[69]:


model.load_state_dict(torch.load('w_sgc.pth'))  

l_pred, l_tar, l_tar_cls, l_feature = [], [], [], []
for i, (inp_ids, inp_mask, inp2, tar_cls, tar) in enumerate(valid_loader): 
    inp_ids, inp_mask, inp2, tar = inp_ids.to(device), inp_mask.to(device), inp2.to(device).long(), tar.to(device)

    inp_ids = inp_ids.squeeze(1)
    inp_mask = inp_mask.squeeze(1)
    inp2 = inp2.squeeze(1)

    with torch.no_grad():
        pred = model.forward(inp_ids, inp_mask, inp2)
        pred = pred.squeeze(1)    

    l_pred.append(pred.detach().cpu().numpy())
    l_tar.append(tar.detach().cpu().numpy())
    l_tar_cls.append(tar_cls.detach().cpu().numpy())
#     l_feature.append(feature.detach().cpu().numpy())


# In[70]:


preds_bert = np.concatenate(l_pred)
labels_cls = np.concatenate(l_tar_cls)  
labels = np.concatenate(l_tar)  

preds_bert, labels


# In[71]:


flat_accuracy(preds_bert, labels)


# In[184]:


flat_accuracy(preds_bert, labels)


# In[136]:


flat_accuracy(preds_bert, labels)


# In[157]:


flat_accuracy(pred_lgbm, labels)


# In[185]:


for i in np.arange(0.1, 1, 0.1):
    flat_accuracy(pred_lgbm*i + preds_bert*(1-i), labels)


# In[158]:


for i in np.arange(0.1, 1, 0.1):
    flat_accuracy(pred_lgbm*i + preds_bert*(1-i), labels)


# In[73]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.clip(preds_bert, 0, 9).round(0), labels)
cm


# In[186]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.clip(preds_bert, 0, 9).round(0), y_valid)
cm


# In[155]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.clip(preds_bert, 0, 9).round(0), y_valid)
cm


# In[156]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.clip(pred_lgbm, 0, 9).round(0), y_valid)
cm


# In[159]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.clip(pred_lgbm*0.1 + preds_bert*0.9, 0, 9).round(0), y_valid)
cm


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[102]:


#0.00005 mse tfX
run(model, train_loader, valid_loader, 50) 


# ### Syn -

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 원본

# In[54]:


from sklearn.model_selection import train_test_split
conts_id_list = df.conts_id.unique()
len(conts_id_list)

train_conts_ids, valid_conts_ids = train_test_split(conts_id_list, test_size=0.3, random_state=1, stratify = df.use_grade)
len(train_conts_ids), len(valid_conts_ids)


# In[55]:


df_train = df[df.conts_id.isin(train_conts_ids)]
df_valid = df[df.conts_id.isin(valid_conts_ids)]


# In[49]:


#df['use_new_grade'] 
from sklearn.preprocessing import RobustScaler
robustScaler = RobustScaler()
robustScaler.fit(df_train[['use_time']])
df_train['use_new_grade'] = robustScaler.transform(df_train[['use_time']])
df_valid['use_new_grade'] = robustScaler.transform(df_valid[['use_time']])


# In[50]:


df_train['use_new_grade'].max()


# In[53]:


df_train.head()


# In[81]:


import random
import re
from nltk import sent_tokenize
from tqdm import tqdm
import albumentations
from albumentations.core.transforms_interface import DualTransform, BasicTransform


class NLPTransform(BasicTransform):
    """ Transform for nlp task."""
    LANGS = {
        'en': 'english',
        'it': 'italian', 
        'fr': 'french', 
        'es': 'spanish',
        'tr': 'turkish', 
        'ru': 'russian',
        'pt': 'portuguese'
    }

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

    def get_sentences(self, text, lang='en'):
        return sent_tokenize(text, self.LANGS.get(lang, 'english'))

class ShuffleSentencesTransform(NLPTransform):
    """ Do shuffle by sentence """
    def __init__(self, always_apply=False, p=0.5):
        super(ShuffleSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        sentences = self.get_sentences(text, lang)
        random.shuffle(sentences)
        return ' '.join(sentences), lang        

class ExcludeDuplicateSentencesTransform(NLPTransform):
    """ Exclude equal sentences """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeDuplicateSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        sentences = []
        for sentence in self.get_sentences(text, lang):
            sentence = sentence.strip()
            if sentence not in sentences:
                sentences.append(sentence)
        return ' '.join(sentences), lang

class ExcludeNumbersTransform(NLPTransform):
    """ exclude any numbers """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeNumbersTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'[0-9]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text, lang

class ExcludeHashtagsTransform(NLPTransform):
    """ Exclude any hashtags with # """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeHashtagsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'#[\S]+\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text, lang

class ExcludeUsersMentionedTransform(NLPTransform):
    """ Exclude @users """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUsersMentionedTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'@[\S]+\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text, lang

class ExcludeUrlsTransform(NLPTransform):
    """ Exclude urls """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUrlsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = re.sub(r'https?\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text, lang



class SwapWordsTransform(NLPTransform):
    """ Swap words next to each other """
    def __init__(self, swap_distance=1, swap_probability=0.1, always_apply=False, p=0.5):
        """  
        swap_distance - distance for swapping words
        swap_probability - probability of swapping for one word
        """
        super(SwapWordsTransform, self).__init__(always_apply, p)
        self.swap_distance = swap_distance
        self.swap_probability = swap_probability
        self.swap_range_list = list(range(1, swap_distance+1))

    def apply(self, data, **params):
        text, lang = data
        words = text.split()
        words_count = len(words)
        if words_count <= 1:
            return text, lang

        new_words = {}
        for i in range(words_count):
            if random.random() > self.swap_probability:
                new_words[i] = words[i]
                continue
    
            if i < self.swap_distance:
                new_words[i] = words[i]
                continue
    
            swap_idx = i - random.choice(self.swap_range_list)
            new_words[i] = new_words[swap_idx]
            new_words[swap_idx] = words[i]

        return ' '.join([v for k, v in sorted(new_words.items(), key=lambda x: x[0])]), lang

class CutOutWordsTransform(NLPTransform):
    """ Remove random words """
    def __init__(self, cutout_probability=0.05, always_apply=False, p=0.5):
        super(CutOutWordsTransform, self).__init__(always_apply, p)
        self.cutout_probability = cutout_probability

    def apply(self, data, **params):
        text, lang = data
        words = text.split()
        words_count = len(words)
        if words_count <= 1:
            return text, lang
        
        new_words = []
        for i in range(words_count):
            if random.random() < self.cutout_probability:
                continue
            new_words.append(words[i])

        if len(new_words) == 0:
            return words[random.randint(0, words_count-1)], lang

        return ' '.join(new_words), lang       


# In[82]:


import nltk


# In[83]:


def transforms():
    return albumentations.Compose([
        ExcludeDuplicateSentencesTransform(p=0.9),  # here not p=1.0 because your nets should get some difficulties        
        ExcludeNumbersTransform(p=0.8),
        ExcludeHashtagsTransform(p=0.5),
        ExcludeUsersMentionedTransform(p=0.9),
        ExcludeUrlsTransform(p=0.9),
        ShuffleSentencesTransform(p=0.8),
        CutOutWordsTransform(p=0.1),
        SwapWordsTransform(p=0.1),
    ])


# In[84]:


MAX_LENGTH = 512 #150 #100  #200
# model_path = '../../project/OContents/xlm-roberta-large'
model_path = '../../project/OContents/bert-base-multilingual-cased'


class ContentDataset(Dataset):
    def __init__(self, df, df_meta, test=False, transforms=transforms()):
        self.transforms = transforms
        self.conts_list = df.conts_id.values
        self.df = df
        self.test = test
#         if test==False: self.label = df['use_time'].values/10e8  
        if test==False: self.label = df['use_grade'].values    
        
        if test==False: self.label_cls = df['use_grade'].values  
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)     
        
    def __getitem__(self, idx):  
        conts_id = self.conts_list[idx]
        #text = self.conts_meta.loc[self.conts_meta.conts_id == conts_id, 'synps_mid_sbst'].values[0]
        text = self.df.loc[self.df.conts_id == conts_id, 'text'].values[0]
        
        #text, _ = self.transforms(data=(text, 'en'))['data']
        
        cate_id = self.df.loc[self.df.conts_id == conts_id, 'cate_id'].values[0]
        mgenre_id = self.df.loc[self.df.conts_id == conts_id, 'mgenre_id'].values[0]
        sgenre_id = self.df.loc[self.df.conts_id == conts_id, 'sgenre_id'].values[0]
        mnfc_yy_id = self.df.loc[self.df.conts_id == conts_id, 'mnfc_yy_id'].values[0]
        
        writer_id_list = self.df.loc[self.df.conts_id == conts_id, 'writer_id'].values[0]
        writer_id = np.zeros(len_writer_seq + 1)
        writer_id[0:len(writer_id_list)] = writer_id_list
        
        dirt_id_list = self.df.loc[self.df.conts_id == conts_id, 'dirt_id'].values[0]
        dirt_id = np.zeros(len_dirt_seq + 1)
        dirt_id[0:len(dirt_id_list)] = dirt_id_list
        
        actor_id_list = self.df.loc[self.df.conts_id == conts_id, 'actor_id'].values[0]
        actor_id = np.zeros(len_actor_seq + 1)
        actor_id[0:len(actor_id_list)] = actor_id_list

        
#         text = "[CLS] " + text + " [SEP]"
        
#         print(text)
#         if self.transforms:
#             text, _ = self.transforms(data=(self.conts_meta[self.conts_list[idx]]['synps_mid_sbst'], 'kr'))['data']   
#         print(text)
        encoded = self.tokenizer.encode_plus(text, 
                                             add_special_tokens=True,
                                             max_length=MAX_LENGTH,
                                             # truncation=True,
                                             pad_to_max_length=True,
                                             return_tensors="pt")
#         print(len(text), (encoded['input_ids']!=0).sum(), int((encoded['input_ids']!=0).sum())/len(text))

        input2 = np.expand_dims(np.concatenate((np.array([cate_id, mgenre_id, sgenre_id, mnfc_yy_id]), writer_id, dirt_id, actor_id), axis = 0), 0)

        if self.test:
            return encoded['input_ids'], encoded['attention_mask'], input2
        
        
#             return encoded['input_ids'], encoded['attention_mask']
        else:
            return encoded['input_ids'], encoded['attention_mask'], input2, self.label_cls[idx], self.label[idx]
#             return encoded['input_ids'], encoded['attention_mask'], self.label[idx]
            

    def __len__(self):
        return len(self.conts_list)


# In[85]:


# tokenizer = AutoTokenizer.from_pretrained('../../project/OContents/xlm-roberta-large')  
# inputs = tokenizer("Hello, my dog is cute", max_length=512, return_tensors="pt")
# inputs['input_ids'], inputs['attention_mask']     
# model = AutoModel.from_pretrained(model_nm)
# output = model(**inputs)
# output['last_hidden_state'].shape
# output['pooler_output'].shape

# #[vocab_[input_id]  for input_id in inputs['input_ids'][0]]

# for input_id in inputs['input_ids'][0].numpy():    
#     vocab_[input_id]


# In[86]:


batch_size = 3 #32 #8 #16 #32
num_workers = 0

train_db = ContentDataset(df_train, df_meta, False)#, transforms=transforms())
valid_db = ContentDataset(df_valid, df_meta, False)

train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_db, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# In[87]:


train_db[0]


# In[88]:


for i, (inp_ids, inp_mask, inp2, tar_cls, tar) in enumerate(train_loader):
    
    break


# In[89]:


inp_ids.shape, inp_mask.shape, inp2.shape


# In[82]:


class ContentModel(nn.Module):  
    def __init__(self, backbone, dropout=0.0, num_class=1):
        super(ContentModel, self).__init__()        
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)        
#         self.w = nn.Sequential(nn.Linear(num_class+16, num_class))

        self.cate_emb = nn.Embedding(len(dic_cate_)+1, 2)
        self.mgenre_emb = nn.Embedding(len(dic_mgenre_)+1, 4)
        self.sgenre_emb = nn.Embedding(len(dic_sgenre_)+1, 4)
        self.mnfc_yy_emb = nn.Embedding(len(dic_mnfc_yy_)+1, 4)
        
        #(배치 크기, 시퀀스 길이) => (배치 크기, 시퀀스 길이, 임베딩 차원)
        
        self.writer_emb = nn.Embedding(len(dic_writer_)+1, 64)
        self.dirt_emb = nn.Embedding(len(dic_dirt_)+1, 64)
        self.actor_emb = nn.Embedding(len(dic_actor_)+1, 64)
        
        self.w = nn.Linear(2*self.backbone.pooler.dense.out_features + (2+4*3+64*3), 512)

        self.w1 = nn.Linear(512, 128)
        self.w2 = nn.Linear(128, 64)
        self.w3 = nn.Linear(64, 16)
#         self.w4 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
        self.w4 = nn.Sequential(nn.Linear(16, 1))
    
        self.v = nn.Linear(self.backbone.pooler.dense.out_features, 1)
    def forward(self, input_ids, attention_mask, x2):        
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)  #x: (batch, seq, emb_size) x2: (batch, emb_size)             
        x_last = output['last_hidden_state'] # (batch, seq_len, dim_bert)
        x_pool = output['pooler_output']  # (batch, dim_bert)    
        return self.v(x_pool)

        cate = self.cate_emb(x2[:, 0])
        mgenre = self.mgenre_emb(x2[:, 1])
        sgenre = self.sgenre_emb(x2[:, 2])
        mnfc_yy = self.mnfc_yy_emb(x2[:, 3])
        
        #len_actor_seq, len_writer_seq, len_dirt_seq

        start_idx = 4
        end_idx = start_idx + len_writer_seq
        writer = self.writer_emb(x2[:, start_idx:end_idx])
        
        start_idx = end_idx
        end_idx = start_idx + len_dirt_seq
        dirt = self.dirt_emb(x2[:, start_idx:end_idx])
        
        start_idx = end_idx
        end_idx = start_idx + len_actor_seq
        actor = self.actor_emb(x2[:, start_idx:end_idx])
    
        x = torch.cat([cate, mgenre, sgenre, mnfc_yy, torch.mean(writer, 1), torch.mean(dirt, 1), torch.mean(actor, 1)], axis=-1)
        x = torch.cat([x, x_pool, torch.mean(x_last, 1)], axis=-1)
        x = self.dropout(x)
        x = self.w(x)
        x = self.dropout(x)
        x = self.w1(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        x = self.w3(x)
        x = self.w4(x)
        
        
        return x


# In[83]:


for i, (inp_ids, inp_mask, inp2, tar_cls, tar) in enumerate(train_loader):
    
    break
# inp_ids, inp_mask, inp2, tar = inp_ids.to(device), inp_mask.to(device), inp2.to(device).long(), tar.to(device)

inp_ids = inp_ids.squeeze(1)
inp_mask = inp_mask.squeeze(1)
inp2 = inp2.long().squeeze(1)


# In[84]:


ContentModel(backbone=AutoModel.from_pretrained(model_path, num_labels = 10) , dropout=0.3, num_class=10)(inp_ids, inp_mask, inp2).shape


# In[93]:


from transformers import BertForSequenceClassification

config = BertConfig.from_json_file(model_path+'/config.json')
model = ContentModel(backbone=AutoModel.from_pretrained(model_path) , dropout=0.3, num_class=10)   

#opt = torch.optim.Adam(model.parameters(), lr=0.00001) #0.000005, betas=(0.9, 0.98), eps=1e-9)

#tfX 
#0.000001 Epoch: 25  Loss: 1.323719  Acc: 0.1148 Acc1: 0.2319 Acc2: 0.2535 Best Epoch: 10 Best Acc: 0.1233

opt = torch.optim.AdamW(model.parameters(), lr=0.000005)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)  


# In[94]:


model


# In[95]:


# tokenizer = AutoTokenizer.from_pretrained('../../project/OContents/bert-base-multilingual-cased') 
# text = '1,600만의 선택! 지금까지 이런 흥행은 없었다! 낮에는 치킨장사, 밤에는 잠복근무. 해체 위기를 맞는 마약반 5인방은 감시를 위해 범죄조직의 아지트 앞 치킨집을 인수해 위장 창업을 하게 되고 치킨집은 맛집으로 입소문 나기 시작한다. 범인을 잡을 것인가, 닭을 잡을 것인가!'
# encoded = tokenizer.encode_plus(text, add_special_tokens=True,
#                                     max_length=MAX_LENGTH,
#                                     return_tensors="pt")


# In[96]:


#[tokenizer.convert_ids_to_tokens(s) for s in encoded['input_ids'].tolist()[0]]


# In[97]:


# 정확도 계산 함수
def flat_accuracy(preds, labels):
    
    pred_flat = preds.flatten().round()
    labels_flat = labels.flatten().astype(int)
    
    acc = np.count_nonzero(pred_flat - labels_flat == 0) / len(labels_flat)
    acc1 = np.count_nonzero(np.abs(pred_flat - labels_flat) == 1) / len(labels_flat)
    acc2 = np.count_nonzero(np.abs(pred_flat - labels_flat) == 2) / len(labels_flat)
    
    return acc, acc1, acc2


# In[98]:


# 정확도 계산 함수
def mse_flat_accuracy(preds, labels):
    
    pred_flat = robustScaler.inverse_transform(np.array(preds.flatten()).reshape(-1, 1))
    pred_flat = [bs.bisect(l_quantile, x) - 1 for x in pred_flat]
    
    labels_flat = labels.flatten().astype(int)
    
    acc = np.count_nonzero(pred_flat - labels_flat == 0) / len(labels_flat)
    acc1 = np.count_nonzero(np.abs(pred_flat - labels_flat) == 1) / len(labels_flat)
    acc2 = np.count_nonzero(np.abs(pred_flat - labels_flat) == 2) / len(labels_flat)
    
    return acc, acc1, acc2


# In[99]:


## loss label smoothing
def run_epoch(dataloaders, model, is_train):
    "Standard Training and Logging Function"
    start = time.time()
    best_acc = 0.0
    l_total_loss, l_total_acc = [], []
    l_pred, l_tar, l_tar_cls = [], [], []
        
    for i, (inp_ids, inp_mask, inp2, tar_cls, tar) in enumerate(dataloaders): 
        inp_ids, inp_mask, inp2, tar = inp_ids.to(device), inp_mask.to(device), inp2.to(device).long(), tar.to(device)
        
#         torch.tensor(b_input_ids).to(device).long()
        inp_ids = inp_ids.squeeze(1)
        inp_mask = inp_mask.squeeze(1)
        inp2 = inp2.squeeze(1)

        with torch.set_grad_enabled(is_train):
            pred = model.forward(inp_ids, inp_mask, inp2).squeeze(1)    
#             loss = torch.nn.L1Loss()(pred.float(), tar.float()) 
            loss = torch.nn.MSELoss()(pred.float(), tar.float()) 
            
#             pred = model.forward(inp_ids, inp_mask, inp2)
#             loss = nn.CrossEntropyLoss()(pred, tar.type(torch.long))
    
            l_total_loss.append(loss.detach().cpu().numpy())

            if is_train:
                loss.backward()
                opt.step()
                opt.zero_grad()
                
                if i % 100 == 1:
                    elapsed = time.time() - start
                    print("Epoch Step: %d Loss: %f Time: %f" %   (i, np.array(l_total_loss).mean(), elapsed))
                    start = time.time()
            else:
                
                l_pred.append(pred.detach().cpu().numpy())
                l_tar.append(tar.detach().cpu().numpy())
                l_tar_cls.append(tar_cls.detach().cpu().numpy())
                
                
    total_loss = np.array(l_total_loss).mean()
    
    if is_train:
        return total_loss
    else:
        y_pred = np.concatenate(l_pred)
        y_test = np.concatenate(l_tar)        

        total_acc, total_acc1, total_acc2 =  flat_accuracy(y_pred, y_test)
        return total_loss, total_acc, total_acc1, total_acc2


# In[100]:


save_path='./model_whole_synp_tf_1e6_5.pth'
def run(model, train_loader, valid_loader, n_epoch=30, save_path='./w_1.pth'):  
    best = {"loss": 1e10, "acc" : 0.0, "epoch" : 0}
    

    for epoch in range(n_epoch):
        t = time.time()
        _ = model.train()
        total_loss = run_epoch(train_loader, model, True)
        print("Epoch: %d  Loss: %f  Time: %f" % (epoch, total_loss, time.time()-t))
        _ = model.eval()
        total_loss, total_acc, total_acc1, total_acc2 = run_epoch(valid_loader, model, False)

        if total_acc > best["acc"]:    
            best["acc"] = total_acc
            best["epoch"] = epoch
            torch.save(model.state_dict(), save_path)

        print("Epoch: %d  Loss: %f  Acc: %.4f Acc1: %.4f Acc2: %.4f Best Epoch: %d Best Acc: %.4f" % (epoch, total_loss, total_acc, total_acc1, total_acc2, best["epoch"], 
                                                                                best["acc"]))
        print('')


# In[101]:


len(train_db), len(valid_db)


# In[102]:


#0.00005 mse tfX
run(model, train_loader, valid_loader, 50) 


# In[102]:


#0.00005 mse tfX
run(model, train_loader, valid_loader, 50) 


# In[ ]:





# In[ ]:





# In[154]:


#0.00005 mse tfX
run(model, train_loader, valid_loader, 50) 


# In[127]:


#0.00001 mse tfX
run(model, train_loader, valid_loader, 50) 


# In[128]:


run(model, train_loader, valid_loader, 50) 


# In[88]:


## torch.set_num_threads(3)
#tf 0.00001
#Epoch: 31  Loss: 2.241614  Acc: 0.1341 Acc1: 0.2327 Acc2: 0.2512 Best Epoch: 26 Best Acc: 0.1364
run(model, train_loader, valid_loader, 50) 


# In[ ]:


## torch.set_num_threads(3)
run(model, train_loader, valid_loader, 50) 


# In[ ]:





# In[213]:


#model.load_state_dict(torch.load(save_path, map_location=device))  

l_pred, l_tar, l_tar_cls = [], [], []
for i, (inp_ids, inp_mask, inp2, tar_cls, tar) in enumerate(valid_loader): 
    inp_ids, inp_mask, inp2, tar = inp_ids.to(device), inp_mask.to(device), inp2.to(device).long(), tar.to(device)

    inp_ids = inp_ids.squeeze(1)
    inp_mask = inp_mask.squeeze(1)
    inp2 = inp2.squeeze(1)

    with torch.no_grad():
        pred = model.forward(inp_ids, inp_mask, inp2).squeeze(1)    

    l_pred.append(pred.detach().cpu().numpy())
    l_tar.append(tar.detach().cpu().numpy())
    l_tar_cls.append(tar_cls.detach().cpu().numpy())

    
    if i == 5:
        break


# In[214]:


preds = np.concatenate(l_pred)
labels_cls = np.concatenate(l_tar_cls)  
labels = np.concatenate(l_tar)  


# In[216]:


preds


# In[217]:


labels


# In[221]:


import matplotlib.pyplot as plt


# In[224]:


plt.plot(labels)
plt.plot(preds)


# In[ ]:


for conts_id in df.conts_id.unique()[:100]:

    print(conts_id, conts_meta[conts_id]['synps_mid_sbst'],"\n")


# In[49]:


get_ipython().system('nvidia-smi')


# In[ ]:





import pandas as pd
from pandas import get_dummies
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import numpy as np
import os
import gc
from multiprocessing import Pool
from joblib import dump, load
import time

import contextlib
import xlearn as xl
t_start = time.time()
user_one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house']
aid_one_hot_feature = ['advertiserId', 'campaignId',  'creativeId', 'adCategoryId', 'productId', 'productType',  'creativeSize']
user_vector_feature = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2', 'marriageStatus', 'os', 'ct']
one_hot_feature =  user_one_hot_feature + aid_one_hot_feature
feature = one_hot_feature + user_vector_feature
user_feature=pd.read_csv('./cache/userFeature.csv')
ad_feature=pd.read_csv('./preliminary_contest_data/adFeature.csv')
print ('fillna')
for col in user_one_hot_feature:
    user_feature[col] = user_feature[col].fillna(-1)
for col in aid_one_hot_feature:
    ad_feature[col] = ad_feature[col].fillna(-1)
for col in user_vector_feature:
    user_feature[col] = user_feature[col].fillna("-1")
print ('fit transform')
feature_index = dict()
last_idx = 0
print ('prepare train test')
train=pd.read_csv('./preliminary_contest_data/train.csv')
train.replace({'label': {-1: 0}}, inplace=True)
test =pd.read_csv('./preliminary_contest_data/test1.csv')
train = pd.merge(train, ad_feature, on='aid', how='left')
y = train.label.values
train = pd.merge(train, user_feature, on='uid', how='left')
test = pd.merge(test, ad_feature, on='aid', how='left')
test = pd.merge(test, user_feature, on='uid', how='left')
filed2feature={}
fileds = 0
for col in feature:
    print(col)
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)
    fuck = train[[col]].values
    for i in fuck:
        if(pd.isnull(i[0])):
            continue
        split_feature = i[0].split(' ')
        for j in split_feature:
            name = '{}_{}'.format(col, j)
            if(name not in feature_index):
                feature_index[name] = last_idx
                filed2feature[name] = fileds
                last_idx+=1
    fileds+=1

file_train = open('./cache/train.ffm','w')
fuck = train[['label']+feature].values
cnt = 0
for i in fuck:
    cnt += 1
    if (cnt % 10000 == 0):
        print(cnt)
    line_str=[str(i[0])]
    for j in range(1,len(i)):
        if(pd.isnull(i[j])):
            continue
        split_feature = i[j].split(' ')
        for k in split_feature:
            name = '{}_{}'.format(feature[j-1], k)
            line_str.append('{}:{}:{}'.format(filed2feature[name],feature_index[name],1))
    file_train.writelines(' '.join(line_str))
    file_train.write("\n")
file_train.close()
file_train = open('./cache/test.ffm','w')
fuck = test[feature].values
cnt = 0
for i in fuck:
    cnt +=1
    line_str=[]
    for j in range(0, len(i)):
        if (pd.isnull(i[j])):
            continue
        split_feature = i[j].split(' ')
        for k in split_feature:
            name = '{}_{}'.format(feature[j], k)
            if(name not in filed2feature):
                continue
            line_str.append('{}:{}:{}'.format(filed2feature[name], feature_index[name], 1))
    if (cnt % 10000 == 0):
        print(cnt,' '.join(line_str))
    file_train.writelines(' '.join(line_str))
    file_train.write("\n")
file_train.close()





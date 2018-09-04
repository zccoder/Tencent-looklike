# coding=utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

import hashlib, csv, math, os, pickle, subprocess

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1

def gen_hashed_fm_feats(feats, nr_bins = int(1e+6)):
    feats = ['{0}:{1}:1'.format(field-1, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats




one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','aid','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['kw1','kw2','kw3','topic1','topic2','topic3']
combine_feature = ['interest1','interest2','interest3','interest4','interest5']

print "reading data"
f = open('../../data/combine_train.csv','rb')
features = f.readline().strip().split(',')
dict = {}
num = 0
for line in f:
    datas = line.strip().split(',')
    for i,d in enumerate(datas):
        if not dict.has_key(features[i]):
            dict[features[i]] = []
        dict[features[i]].append(d)
    num += 1

f.close()

print "transforming data"
ftrain =  open('../../data/train.ffm','wb')

for i in range(num):
    feats = []
    for j, f in enumerate(one_hot_feature,1):
        field = j
        feats.append((field, f+'_'+dict[f][i]))

    for j, f in enumerate(vector_feature,1):
        field = j + len(one_hot_feature)
        xs = dict[f][i].split(' ')
        for x in xs:
            feats.append((field, f+'_'+x))

    for j, f in enumerate(combine_feature,1):
        field = j + len(one_hot_feature) + len(vector_feature)
        xs = dict[f][i].split(' ')
        for x in xs:
            feats.append((field, 'aid_'+dict['aid'][i]+'_'+f+'_'+x))

    feats = gen_hashed_fm_feats(feats)
    ftrain.write(dict['label'][i] + ' ' + ' '.join(feats) + '\n')


ftrain.close()


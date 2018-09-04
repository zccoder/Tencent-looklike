import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc
import numpy as np
import time
import sys


t1=time.time()
ad_feature=pd.read_csv('adFeature.csv')
user_feature=pd.read_csv('userFeature.csv')
print('1all time:',time.time()-t1)
train=pd.read_csv('train.csv')
predict1=pd.read_csv('test1.csv')
predict2=pd.read_csv('test2.csv')
predict=pd.concat([predict1,predict2])

train.loc[train['label']==-1,'label']=0
predict['label']=-1
data=pd.concat([train,predict])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
del ad_feature,user_feature;gc.collect()
print('2all time:',time.time()-t1)

save_feature=['aid','uid','label','interest1']

data=data[save_feature]
data=data.fillna('0 0')
jiaocha_feature=data[['aid','uid','label']]

print('3all time:',time.time()-t1)
def field_merge(x,y):
    cross=""
    x=str(x)
    y=str(y).split()
    for i in y:
         cross+=x+"_"+i+" "
    return cross.strip()

import random
def tran(x,_max_length):
  x=x.split(" ")
  for i in range(len(x)):
    x[i]=int(x[i])
  random.shuffle(x)
  if(len(x)<_max_length):
    for i in range(_max_length-len(x)):
      x.append(0)
  if(len(x)>_max_length):
    for i in range(len(x)-_max_length):
      x.pop()   
  return np.array(x)	
	
	
hand_main_vector=['aid']


hand_other_vector=['interest1']
_max_length=33
import gc
print('begin.....')
for main_fe in hand_main_vector:
   for other_afe in hand_other_vector:
      jiaocha_feature[main_fe+'_'+other_afe]=data[[main_fe,other_afe]].apply(lambda x:field_merge(x[main_fe],x[other_afe]),axis=1)
      print('xxall time:',time.time()-t1)
      jiaocha_feature[main_fe+'_'+other_afe]=jiaocha_feature[main_fe+'_'+other_afe].apply(lambda x:tran(x,_max_length))


print('4all time:',time.time()-t1)
print(jiaocha_feature.head())
one_hot_feature=[]
for i in range(_max_length):
   print('i',i,time.time()-t1)
   one_hot_feature.append("col"+str(i+1))
   jiaocha_feature["col"+str(i+1)]=jiaocha_feature[main_fe+'_'+other_afe].apply(lambda x:x[i])
print(jiaocha_feature.head())
print('jiaocha_feature.shape',jiaocha_feature.shape)
jiaocha_feature.drop(main_fe+'_'+other_afe,axis=1,inplace=True)  
print(jiaocha_feature.head())
print('jiaocha_feature.shape',jiaocha_feature.shape)
for feature in one_hot_feature:
    try:
        jiaocha_feature[feature] = LabelEncoder().fit_transform(jiaocha_feature[feature].apply(int))
    except:
        jiaocha_feature[feature] = LabelEncoder().fit_transform(jiaocha_feature[feature])
    print(feature,jiaocha_feature[feature].max(),time.time()-t1)

print('jiaocha_feature.shape',jiaocha_feature.shape)
jiaocha_feature.to_csv("./SDD_data/final_linshi_sdd_handcraft_field_embedding_feature_aid_insterest1.csv",header=True,index=False)

print('5all time:',time.time()-t1)
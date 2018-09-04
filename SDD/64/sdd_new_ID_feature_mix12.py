import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import numpy as np
import time
t1=time.time()
ad_feature=pd.read_csv('adFeature.csv')
user_feature=pd.read_csv('userFeature.csv')
print('all time:',time.time()-t1)
train=pd.read_csv('train.csv')
predict1=pd.read_csv('test1.csv')
predict2=pd.read_csv('test2.csv')
predict=pd.concat([predict1,predict2])
train.loc[train['label']==-1,'label']=0
predict['label']=-1
data=pd.concat([train,predict])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
data=data.fillna('-1')
save_feature=['aid','uid','label','LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType','creativeSize']

data=data[save_feature]
one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType','creativeSize']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

single_emb=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId','productType','creativeSize']
#singel_max=[853,5,3,2,7,2,1,4,64,26,78,137,172,39,32,3,14]

for i in single_emb:
   print(data[i].max())

data.to_csv("./SDD_data/final_sdd_single_onehot_embedding_feature2_mix_test12.csv",header=True,index=False)


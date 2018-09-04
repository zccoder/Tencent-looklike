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
choose_index=int(sys.argv[1])
print('choose_index:',choose_index)
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
data=data.fillna('-1')
save_feature=['aid','uid','label','LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType','creativeSize']

data=data[save_feature]

jiaocha_feature=data[['aid','uid','label']]

print('3all time:',time.time()-t1)
def field_merge(x,y):
    cross=""
    x=str(x)
    y=str(y).split()
    for i in y[:-1]:
         cross+=x+"_"+i+" "
    cross+=x+"_"+y[-1]
    return cross.strip()

hand_main_vector=['aid']

if(choose_index==0):   
  hand_other_vector=['age','carrier','consumptionAbility','education','gender','house','os','ct']
elif(choose_index==1):
  hand_other_vector=['marriageStatus','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType','creativeSize']
	   
	   
print('begin.....')
for main_fe in hand_main_vector:
   for other_afe in hand_other_vector:
      jiaocha_feature[main_fe+'_'+other_afe]=data[[main_fe,other_afe]].apply(lambda x:str(x[main_fe])+'_'+str(x[other_afe]),axis=1)
      try:
     	  jiaocha_feature[main_fe+'_'+other_afe]=LabelEncoder().fit_transform(jiaocha_feature[main_fe+'_'+other_afe].apply(int))
      except:
     	  jiaocha_feature[main_fe+'_'+other_afe]=LabelEncoder().fit_transform(jiaocha_feature[main_fe+'_'+other_afe])	  
      print(main_fe+'_'+other_afe,jiaocha_feature[main_fe+'_'+other_afe].max())
      print('xxall time:',time.time()-t1)

print('4all time:',time.time()-t1)
if(choose_index==0):  
  jiaocha_feature.to_csv("./SDD_data/final_sdd_handcraft_onehot_embedding_feature3.csv",header=True,index=False)
elif(choose_index==1):  
  jiaocha_feature.to_csv("./SDD_data/final_sdd_handcraft_onehot_embedding_feature4.csv",header=True,index=False)
print('5all time:',time.time()-t1)
#encoding=utf-8
"""
 1 appIdAction,537,6217
  2 appIdInstall,920,64869
  3 interest1,47,122
  4 interest2,32,82
  5 interest3,10,10
  6 interest4,10,10
  7 interest5,86,136
  8 kw1,5,796741
  9 kw2,5,121958
 10 kw3,5,58782
 11 topic1,5,9999
 12 topic2,5,9999
 13 topic3,5,9463
"""


import numpy as np
import pandas as pd
import time
import gc
import sys

index=int(sys.argv[1])
#index=8
sparse_col_name=['appIdAction','appIdInstall','interest1','interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3',
               'topic1', 'topic2', 'topic3']
#sparse_col_max_length=[537,920,47,32,10,10,86,5,5,5,5,5,5]
sparse_col_true_max_length=[33,33,33,-1,10,10,-1,-1,-1,-1,-1,-1,-1]
'''
全部数据 77794891
3:  interest1  maxlen 33  maxvalue  122  ok!
4:  interest2  maxlen 33  maxvalue  82   ok!
5:  interest3  maxlen 10  maxvalue  10  ok!
6:  interest4  maxlen 10  maxvalue  10  ok!  
7:  interest5  maxlen 33  maxvalue  136  ok!  
8:  kw1        maxlen 5   maxvalue  796741  ok!
9:  kw2        maxlen 5   maxvalue  121958  ok!
10: kw3        maxlen 5   maxvalue  58782  ok!
11: topic1     maxlen 5  maxvalue  9999  ok!
12: topic2     maxlen 5  maxvalue  9999  ok!
13: topic3     maxlen 5  maxvalue  9463  ok!
'''
chusai_addr='/home/flypiggy/Downloads/guorenhe/tencent'

max_length_limit=33


t1=time.time()
def tran(x,_max_length):
  x=x.split(" ")
  for i in range(len(x)):
    x[i]=int(x[i])
  if(len(x)<_max_length):
    for i in range(_max_length-len(x)):
      x.append(0)
  if(len(x)>_max_length):
    for i in range(len(x)-_max_length):
      x.pop()   
  return np.array(x)

#--------------------------------------------------------------------------------
#user_feature_chusai=pd.read_csv('%s/userFeature.csv'%chusai_addr)#chusai
#user_feature1=pd.read_csv('userFeature1.csv')#fusai 1
#user_feature2=pd.read_csv('userFeature2.csv')#fusai 2

#aa=pd.concat([user_feature1,user_feature2,user_feature_chusai])
#del user_feature_chusai,user_feature1,user_feature2;gc.collect()
#aa=aa.drop_duplicates(subset=['uid'])


aa=pd.read_csv('userFeature.csv')#mix userfeature
print('aa shape',aa.shape)
print('index:',index,"col name:",sparse_col_name[index-1],"time",time.time()-t1)

user_feature=pd.concat([aa["uid"],aa[sparse_col_name[index-1]]],axis=1)#
del aa
gc.collect()
#--------------------------------------------------------------------------------

#ad_feature_fusai=pd.read_csv('adFeature.csv')#fusai
#ad_feature_chusai=pd.read_csv('%s/adFeature.csv'%chusai_addr)#chusai
#ad_feature=pd.concat([ad_feature_fusai,ad_feature_chusai])
#del ad_feature_fusai,ad_feature_chusai;gc.collect()
#--------------------------------------------------------------------------------


train=pd.read_csv("train.csv")
#--------------------------------------------------------------------------------
predict1=pd.read_csv('test1.csv')
predict2=pd.read_csv('test2.csv')
predict=pd.concat([predict1,predict2])
del predict1,predict2;gc.collect()
#--------------------------------------------------------------------------------

print("read prepared!")
train.loc[train['label']==-1,'label']=0

predict['label']=-1

data=pd.concat([train,predict])

#data=pd.merge(data,ad_feature,on='aid',how='left')

data=pd.merge(data,user_feature,on='uid',how='left')
print("merge over!")


data[sparse_col_name[index-1]]=data[sparse_col_name[index-1]].fillna('0 0')

#--------------------------------------------------------------------------------
#_max_length=sparse_col_max_length[index-1]

if(sparse_col_true_max_length[index-1]>0):
  _max_length=sparse_col_true_max_length[index-1]
elif(sparse_col_true_max_length[index-1]<0):
  print('begin to compute the max length',time.time()-t1)
  _temp_col=data[sparse_col_name[index-1]].apply(lambda x:len(x.split(" ")))
  _max_length=_temp_col.max()
  del _temp_col
  gc.collect()
#--------------------------------------------------------------------------------
print('maxlength:',_max_length,"time",time.time()-t1)
if(_max_length>max_length_limit):
  _max_length=max_length_limit
  print('the maxlength exceeds the maximum limit',_max_length,"time",time.time()-t1)

data[sparse_col_name[index-1]]=data[sparse_col_name[index-1]].apply(lambda x:tran(x,_max_length))
_temp_col=data[sparse_col_name[index-1]].apply(lambda x:x.max())
_max_value=_temp_col.max()
print("max_value:",_max_value)
del _temp_col
gc.collect()
#--------------------------------------------------------------------------------
print('ori data shape:',data.shape,"time",time.time()-t1)
#print(data.head())
for i in range(_max_length):
   print('i',i,time.time()-t1)
   data["col"+str(i+1)]=data[sparse_col_name[index-1]].apply(lambda x:x[i])
print('new data shape1:',data.shape,"time",time.time()-t1)
#print(data.head())  
data.drop(sparse_col_name[index-1],axis=1,inplace=True)  
print('new data shape2:',data.shape,"time",time.time()-t1)

data.to_csv("./SDD_data/final_sdd_embedding_feature_mix_chusai_%s.csv"%sparse_col_name[index-1],header=True,index=False)




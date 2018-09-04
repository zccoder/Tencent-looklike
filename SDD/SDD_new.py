"""

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
sparse_col_max_length=[537,920,47,32,10,10,86,5,5,5,5,5,5]

t1=time.time()
def tran(x,_max_length):
  x=x.split(" ")
  for i in range(len(x)):
    x[i]=int(x[i])
  if(len(x)<_max_length):
    for i in range(_max_length-len(x)):
      x.append(0)
  return np.array(x)


aa=pd.read_csv("userFeature.csv")
print('index:',index,"col name:",sparse_col_name[index-1],"time",time.time()-t1)

user_feature=pd.concat([aa["uid"],aa[sparse_col_name[index-1]]],axis=1)#
del aa
gc.collect()

ad_feature=pd.read_csv("adFeature.csv")
train=pd.read_csv("train.csv")
predict=pd.read_csv("test2.csv")
print("read prepared!")
train.loc[train['label']==-1,'label']=0
predict['label']=-1
data=pd.concat([train,predict])
#data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
print("merge over!")


data[sparse_col_name[index-1]]=data[sparse_col_name[index-1]].fillna('0 0')


_max_length=sparse_col_max_length[index-1]
print('maxlength:',_max_length,"time",time.time()-t1)
data[sparse_col_name[index-1]]=data[sparse_col_name[index-1]].apply(lambda x:tran(x,_max_length))
_temp_col=data[sparse_col_name[index-1]].apply(lambda x:x.max())
_max_value=_temp_col.max()
print("max_value:",_max_value)
del _temp_col
gc.collect()

print('ori data shape:',data.shape,"time",time.time()-t1)
print(data.head())
for i in range(_max_length):
   print('i',i,time.time()-t1)
   data["col"+str(i+1)]=data[sparse_col_name[index-1]].apply(lambda x:x[i])
print('new data shape1:',data.shape,"time",time.time()-t1)
print(data.head())  
data.drop(sparse_col_name[index-1],axis=1,inplace=True)  
print('new data shape2:',data.shape,"time",time.time()-t1)
print(data.head())  
data.to_csv("./embedding_data/sdd_embedding_feature_%s.csv"%sparse_col_name[index-1],header=True,index=False)




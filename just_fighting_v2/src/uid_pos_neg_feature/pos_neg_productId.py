import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
from datetime import datetime
#此处只使用了uid和productId的正负
#数据读取
t1=datetime.now()
print("reading..")
input_path='../../data/'
save_path='../../all_feature/uid_pos_neg_file/'
data=pd.read_csv(input_path+'train_test_data.csv',usecols=['uid','productId','label'])

train=data[data.label!=-1]
test=data[data.label==-1]
del data
gc.collect()

#将所有的uid保存的posproductId和negproductId到字典中，初始化为空
uid_all={}
for i in train.uid.value_counts().index:
	uid_all[str(i)+'_productId_pos']=[]
	uid_all[str(i)+'_productId_neg']=[]
#每行读入，训练集中的pos和neg保存到字典中
#其中i[0] productId i[1] uid i[2] label
temp_train=train[['uid','productId','label']]
for i in temp_train.values:
	if i[2]==1:
		uid_all[str(i[0])+'_productId_pos'].append(str(i[1]))
	else:
		uid_all[str(i[0])+'_productId_neg'].append(str(i[1]))

#对productId_pos和productId_neg进行merge
userFeature_data=[]
for i in temp_train.uid.value_counts().index:
	userFeature_dict={}
	userFeature_dict['uid']=i
	userFeature_dict['productId_pos']=' '.join(uid_all[str(i)+'_productId_pos'])
	userFeature_dict['productId_neg']=' '.join(uid_all[str(i)+'_productId_neg'])
	userFeature_data.append(userFeature_dict)
userFeature_data=pd.DataFrame(userFeature_data)

test=pd.merge(test,userFeature_data,how='left',on='uid')
test[['uid','productId_neg','productId_pos']].to_csv(save_path+"test_neg_pos_productId.csv",index=None)
temp_train=pd.merge(temp_train,userFeature_data,how='left',on='uid')

#print(train)
userFeature_train_data=[]
for i in temp_train.values:
	userFeature_dict={}
	uid=i[0]
	productId=i[1]
	label=i[2]
	productId_neg=i[3]
	productId_pos=i[4]
	userFeature_dict['productId']=productId
	userFeature_dict['label']=label
	userFeature_dict['uid']=uid
	if label==1:
		alllist=productId_pos.split()
		alllist.remove(str(productId))
		productId_pos=' '.join(alllist)
	else:
		alllist=productId_neg.split()
		alllist.remove(str(productId))
		productId_neg=' '.join(alllist)
	userFeature_dict['productId_neg']=productId_neg
	userFeature_dict['productId_pos']=productId_pos
	userFeature_train_data.append(userFeature_dict)
userFeature_train_data=pd.DataFrame(userFeature_train_data)
#print(userFeature_train_data)
train_data=userFeature_train_data
train_data[['uid','productId_neg','productId_pos']].to_csv(save_path+"train_neg_pos_productId.csv",index=None)
print("pos_neg_productId finished")
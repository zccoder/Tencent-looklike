import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
from datetime import datetime
#此处只使用了uid和adCategoryId的正负
#数据读取
t1=datetime.now()
print("reading..")
input_path='../../data/'
save_path='../../all_feature/uid_pos_neg_file/'
data=pd.read_csv(input_path+'train_test_data.csv',usecols=['uid','adCategoryId','label'])

train=data[data.label!=-1]
test=data[data.label==-1]
del data
gc.collect()

#将所有的uid保存的posadCategoryId和negadCategoryId到字典中，初始化为空
uid_all={}
for i in train.uid.value_counts().index:
	uid_all[str(i)+'_adCategoryId_pos']=[]
	uid_all[str(i)+'_adCategoryId_neg']=[]
#每行读入，训练集中的pos和neg保存到字典中
#其中i[0] adCategoryId i[1] uid i[2] label
temp_train=train[['uid','adCategoryId','label']]
for i in temp_train.values:
	if i[2]==1:
		uid_all[str(i[0])+'_adCategoryId_pos'].append(str(i[1]))
	else:
		uid_all[str(i[0])+'_adCategoryId_neg'].append(str(i[1]))

#对adCategoryId_pos和adCategoryId_neg进行merge
userFeature_data=[]
for i in temp_train.uid.value_counts().index:
	userFeature_dict={}
	userFeature_dict['uid']=i
	userFeature_dict['adCategoryId_pos']=' '.join(uid_all[str(i)+'_adCategoryId_pos'])
	userFeature_dict['adCategoryId_neg']=' '.join(uid_all[str(i)+'_adCategoryId_neg'])
	userFeature_data.append(userFeature_dict)
userFeature_data=pd.DataFrame(userFeature_data)

test=pd.merge(test,userFeature_data,how='left',on='uid')
test[['uid','adCategoryId_neg','adCategoryId_pos']].to_csv(save_path+"test_neg_pos_adCategoryId.csv",index=None)
temp_train=pd.merge(temp_train,userFeature_data,how='left',on='uid')

#print(train)
userFeature_train_data=[]
for i in temp_train.values:
	userFeature_dict={}
	uid=i[0]
	adCategoryId=i[1]
	label=i[2]
	adCategoryId_neg=i[3]
	adCategoryId_pos=i[4]
	userFeature_dict['adCategoryId']=adCategoryId
	userFeature_dict['label']=label
	userFeature_dict['uid']=uid
	if label==1:
		alllist=adCategoryId_pos.split()
		alllist.remove(str(adCategoryId))
		adCategoryId_pos=' '.join(alllist)
	else:
		alllist=adCategoryId_neg.split()
		alllist.remove(str(adCategoryId))
		adCategoryId_neg=' '.join(alllist)
	userFeature_dict['adCategoryId_neg']=adCategoryId_neg
	userFeature_dict['adCategoryId_pos']=adCategoryId_pos
	userFeature_train_data.append(userFeature_dict)
userFeature_train_data=pd.DataFrame(userFeature_train_data)
#print(userFeature_train_data)
train_data=userFeature_train_data
train_data[['uid','adCategoryId_neg','adCategoryId_pos']].to_csv(save_path+"train_neg_pos_adCategoryId.csv",index=None)
print("pos_neg_adCategoryId finished")
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import gc
import os
from datetime import datetime
import collections
import itertools
from tqdm import tqdm

t1=datetime.now()
input_path='../../data/'
save_path='../../all_feature/transfer_feature_file/'
usecols=['marriageStatus','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','os','ct','appIdAction','appIdInstall','label']
feat_List=usecols[:-1]
train_test=pd.read_csv(input_path+"train_test_data.csv",usecols=usecols)
train_test=train_test.fillna('-1')
train=train_test[train_test.label!=-1]
test=train_test[train_test.label==-1]
del train_test
gc.collect()

print(train.shape)
print(test.shape)

def func(x_key_list,fea_count_dict,fea_click_count_dict): ##返回点击的平均值，浏览的平均值，转换率平均值
	x_key_list=str(x_key_list).split()
	count_times=30.0001
	click_count_times=1.0001
	click_rate=0.0
	for i in x_key_list:
		rate=100*float((click_count_times+fea_click_count_dict[str(i)])/(count_times+fea_count_dict[str(i)]))
		click_rate+=rate
	click_rate=click_rate/len(x_key_list)
	click_rate=np.round(click_rate,4)
	return click_rate

def get_fea_click_count_dict(train,feat):#
	fea_count=train[feat].apply(lambda x:x.split(' ')).values
	fea_click_count=train[train.label==1][feat].apply(lambda x:x.split(' ')).values
	fea_count_out = list(itertools.chain.from_iterable(fea_count))
	fea_click_count_out=list(itertools.chain.from_iterable(fea_click_count))
	fea_count_dict=collections.Counter(fea_count_out)
	fea_click_count_dict=collections.Counter(fea_click_count_out)
	return fea_count_dict,fea_click_count_dict

def Feature(train,predict,feat):
	train['index']=list(range(train.shape[0]))
	predict['index']=list(range(predict.shape[0]))
	df_stas_feat=None
	kf=KFold(n_splits=5,random_state=2018,shuffle=True)
	for train_index,val_index in kf.split(train):
		X_train=train.loc[train_index,:]
		X_val=train.loc[val_index,:]
		fea_count_dict,fea_click_count_dict=get_fea_click_count_dict(X_train,feat)   
		X_val[feat+'_stas']=X_val[feat].apply(lambda x:func(x,fea_count_dict,fea_click_count_dict))
		df_stas_feat=pd.concat([df_stas_feat,X_val[[feat+'_stas','index']]],axis=0)
	train=pd.merge(train,df_stas_feat,how='left',on='index')
	fea_count_dict,fea_click_count_dict=get_fea_click_count_dict(train,feat)
	predict[feat+'_stas']=predict[feat].apply(lambda x:func(x,fea_count_dict,fea_click_count_dict))
	return train[[feat+'_stas']],predict[[feat+'_stas']]

all_train=None
all_test=None
for feat in tqdm(feat_List):
	train_feat=train[[feat,'label']]
	test_feat=test[[feat]]
	train_,test_=Feature(train_feat,test_feat,feat)
	all_train=pd.concat([all_train,train_],axis=1)
	all_test=pd.concat([all_test,test_],axis=1)
	del train[feat],test[feat]
	gc.collect()
	print(feat,'done!')

all_train.to_csv(save_path+"vector_statis_feature_train.csv",index=None)
all_test.to_csv(save_path+"vector_statis_feature_test.csv",index=None)
t2=datetime.now()
print(t2-t1)
	

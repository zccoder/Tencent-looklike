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
usecols=['creativeSize_kw2_topic2_stas', 'aid_kw2_age_stas', 'creativeSize_topic2_LBS_stas', 'creativeSize_kw2_LBS_stas', 'aid_kw2_topic2_stas', 'aid_age_education_stas', 'creativeSize_age_LBS_stas', 'creativeSize_LBS_education_stas', 'aid_kw2_education_stas', 'aid_age_LBS_stas', 'creativeSize_kw2_age_stas', 'aid_topic2_age_stas', 'aid_topic2_education_stas', 'creativeSize_topic2_age_stas', 'aid_LBS_education_stas', 'aid_topic2_interest1_stas', 'aid_interest1_age_stas', 'aid_kw2_LBS_stas', 'aid_kw2_interest1_stas', 'creativeSize_kw2_education_stas', 'creativeSize_topic2_education_stas', 'aid_interest1_LBS_stas', 'creativeSize_age_education_stas','label']
usecols=[i.replace('stas','stats') for i in usecols]
useTruecols=usecols[4:5]
useTruecols.append(usecols[-1])
feat_List=useTruecols[:-1]

# train=pd.read_csv("cross_id_content/train_three_multi_1.csv",usecols=useTruecols)
# print(train.info(memory_usage='deep'))
# test=pd.read_csv("cross_id_content/test_three_multi_1.csv",usecols=useTruecols)
# print(train.shape)

train=pd.read_csv("aid_kw2_topic2_multi_stas_train.csv")
test=pd.read_csv("aid_kw2_topic2_multi_stas_test.csv")


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

all_train.to_csv("three_multi_statis_feature_train_1_1_5.csv",index=None)
all_test.to_csv("three_multi_statis_feature_test_1_1_5.csv",index=None)

t2=datetime.now()
print(t2-t1)
	

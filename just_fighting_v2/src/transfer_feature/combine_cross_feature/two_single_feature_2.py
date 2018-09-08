import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
from datetime import datetime

t1=datetime.now()
input_path='../../../data/'
save_path='../../../all_feature/combine_feature_file/'
all_cross=['creativeSize_LBS_stas', 'productType_LBS_stas', 'adCategoryId_LBS_stas', 'advertiserId_LBS_stas', 'aid_LBS_stas', 'productId_LBS_stas', 'advertiserId_consumptionAbility_stas', 'campaignId_LBS_stas', 'aid_consumptionAbility_stas', 'aid_gender_stas', 'adCategoryId_age_stas', 'creativeSize_age_stas', 'campaignId_gender_stas', 'aid_carrier_stas', 'creativeSize_gender_stas', 'adCategoryId_consumptionAbility_stas', 'aid_house_stas', 'aid_age_stas', 'advertiserId_gender_stas']
usecols=['aid', 'LBS', 'productType', 'carrier', 'gender', 'age', 'advertiserId', 'house', 'campaignId', 'adCategoryId', 'creativeSize', 'consumptionAbility', 'productId','label']
train_test=pd.read_csv(input_path+"train_test_data.csv",usecols=usecols)
train_test=train_test.fillna('-1')
train=train_test[train_test.label!=-1]
test=train_test[train_test.label==-1]
del train_test
gc.collect()

def hebing(x,y):
	cross=""
	x=str(x)
	y=str(y)
	return x+"_"+y

all_train=pd.DataFrame()
all_test=pd.DataFrame()
for i in tqdm(all_cross[10:]):
	x_split=i.split('_')
	a=x_split[0]
	b=x_split[1]
	new_fea_name=str(a)+'_'+str(b)+'_stats'
	all_train[new_fea_name]=train[[a,b]].apply(lambda x:hebing(x[a],x[b]),axis=1)
	all_test[new_fea_name]=test[[a,b]].apply(lambda x:hebing(x[a],x[b]),axis=1)
	print(new_fea_name,'done')
all_train['label']=train.label
all_test['label']=test.label
all_train.to_csv(save_path+"train_two_single_2.csv",index=None)
all_test.to_csv(save_path+"test_two_single_2.csv",index=None)
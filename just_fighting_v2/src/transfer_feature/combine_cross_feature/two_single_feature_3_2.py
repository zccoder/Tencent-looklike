import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
from datetime import datetime

t1=datetime.now()
input_path='../../../data/'
save_path='../../../all_feature/combine_feature_file/'
all_cross=['campaignId_age_stas', 'advertiserId_house_stas','campaignId_ct_stas','aid_os_stas','adCategoryId_ct_stas','advertiserId_age_stas','adCategoryId_carrier_stas','campaignId_consumptionAbility_stas','productType_age_stas','creativeSize_consumptionAbility_stas','creativeSize_carrier_stas','adCategoryId_gender_stas','advertiserId_carrier_stas','adCategoryId_house_stas','label']
usecols=['ct', 'advertiserId', 'carrier', 'aid', 'os', 'age', 'consumptionAbility', 'house', 'gender', 'adCategoryId', 'creativeSize', 'productType', 'campaignId','label']
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
for i in tqdm(all_cross[5:10]):
	x_split=i.split('_')
	a=x_split[0]
	b=x_split[1]
	new_fea_name=str(a)+'_'+str(b)+'_stats'
	all_train[new_fea_name]=train[[a,b]].apply(lambda x:hebing(x[a],x[b]),axis=1)
	all_test[new_fea_name]=test[[a,b]].apply(lambda x:hebing(x[a],x[b]),axis=1)
	print(new_fea_name,'done')
all_train['label']=train.label
all_test['label']=test.label
all_train.to_csv(save_path+"train_two_single_3_2.csv",index=None)
all_test.to_csv(save_path+"test_two_single_3_2.csv",index=None)
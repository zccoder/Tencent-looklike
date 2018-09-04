import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
from datetime import datetime

all_cross=['productType_kw2_multi_stas', 'creativeSize_kw2_multi_stas', 'creativeSize_topic1_multi_stas', 'adCategoryId_kw2_multi_stas', 'productType_topic2_multi_stas', 'aid_interest2_multi_stas', 'creativeSize_topic2_multi_stas', 'creativeSize_kw1_multi_stas', 'aid_topic2_multi_stas', 'adCategoryId_topic2_multi_stas', 'aid_kw2_multi_stas', 'aid_kw1_multi_stas', 'advertiserId_topic2_multi_stas', 'advertiserId_kw2_multi_stas', 'adCategoryId_kw1_multi_stas', 'campaignId_topic2_multi_stas', 'productType_topic1_multi_stas', 'aid_marriageStatus_stas', 'productId_topic2_multi_stas', 'aid_ct_stas', 'campaignId_kw1_multi_stas', 'aid_topic1_multi_stas', 'adCategoryId_topic1_multi_stas', 'creativeSize_ct_stas', 'productType_kw1_multi_stas', 'advertiserId_kw1_multi_stas', 'productId_kw2_multi_stas', 'advertiserId_topic1_multi_stas', 'creativeSize_marriageStatus_stas', 'productId_topic1_multi_stas', 'campaignId_kw2_multi_stas', 'creativeSize_marriageStatus_multi_stas', 'productId_kw1_multi_stas', 'campaignId_topic1_multi_stas', 'creativeSize_interest2_multi_stas', 'advertiserId_ct_stas', 'productType_interest2_multi_stas', 'adCategoryId_marriageStatus_stas']
#38个分成3组,13,13,12
usecols=['creativeSize', 'topic2', 'campaignId', 'productType', 'marriageStatus', 'interest2', 'topic1', 'advertiserId', 'adCategoryId', 'kw2', 'aid', 'kw1', 'productId', 'ct','label']
train_test=pd.read_csv("../data/train_test1_2_data.csv",usecols=usecols)
train_test=train_test.fillna('-1')
train=train_test[train_test.label!=-1]
test=train_test[train_test.label==-1]
del train_test
gc.collect()

def hebing(x,y):
	cross=""
	x=str(x)
	y=str(y).split()
	for i in y:
		cross+=x+"_"+i+" "
	return cross.strip()

all_train=pd.DataFrame()
all_test=pd.DataFrame()
for i in all_cross[13:26]:
	x_split=i.split('_')
	a=x_split[0]
	b=x_split[1]
	new_fea_name=str(a)+'_'+str(b)+'_stats'
	all_train[new_fea_name]=train[[a,b]].apply(lambda x:hebing(x[a],x[b]),axis=1)
	all_test[new_fea_name]=test[[a,b]].apply(lambda x:hebing(x[a],x[b]),axis=1)
	print(new_fea_name,'done')
all_train['label']=train.label
all_test['label']=test.label
all_train.to_csv("cross_id_content/train_two_multi_2.csv",index=None)
all_test.to_csv("cross_id_content/test_two_multi_2.csv",index=None)
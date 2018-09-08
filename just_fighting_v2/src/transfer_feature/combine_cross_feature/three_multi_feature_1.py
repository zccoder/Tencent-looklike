import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
from datetime import datetime

t1=datetime.now()
input_path='../../../data/'
save_path='../../../all_feature/combine_feature_file/'
all_cross=['creativeSize_kw2_topic2_multi_stas', 'aid_kw2_age_multi_stas', 'creativeSize_topic2_LBS_multi_stas', 'creativeSize_kw2_LBS_multi_stas', 'aid_kw2_topic2_multi_stas', 'aid_age_education_multi_stas', 'creativeSize_age_LBS_multi_stas', 'creativeSize_LBS_education_multi_stas', 'aid_kw2_education_multi_stas', 'aid_age_LBS_multi_stas', 'creativeSize_kw2_age_multi_stas', 'aid_topic2_age_multi_stas', 'aid_topic2_education_multi_stas', 'creativeSize_topic2_age_multi_stas', 'aid_LBS_education_multi_stas', 'aid_topic2_interest1_multi_stas', 'aid_interest1_age_multi_stas', 'aid_kw2_LBS_multi_stas', 'aid_kw2_interest1_multi_stas', 'creativeSize_kw2_education_multi_stas', 'creativeSize_topic2_education_multi_stas', 'aid_interest1_LBS_multi_stas', 'creativeSize_age_education_multi_stas']
usecols=['creativeSize', 'kw2', 'LBS', 'age', 'topic2', 'aid', 'education', 'interest1','label']
train_test=pd.read_csv(input_path+"train_test_data.csv",usecols=usecols)
train_test=train_test.fillna('-1')
train=train_test[train_test.label!=-1]
test=train_test[train_test.label==-1]
del train_test
gc.collect()

def hebing(x,y,z):
	cross=""
	x=str(x)
	y=str(y).split()
	z=str(z).split()
	for i in y:
		for j in z:
			cross+=x+"_"+i+"_"+j+" "
	return cross.strip()

all_train=pd.DataFrame()
all_test=pd.DataFrame()
for i in tqdm(all_cross[0:11]):
	x_split=i.split('_')
	a=x_split[0]
	b=x_split[1]
	c=x_split[2]
	new_fea_name=str(a)+'_'+str(b)+'_'+str(c)+'_stats'
	all_train[new_fea_name]=train[[a,b,c]].apply(lambda x:hebing(x[a],x[b],x[c]),axis=1)
	all_test[new_fea_name]=test[[a,b,c]].apply(lambda x:hebing(x[a],x[b],x[c]),axis=1)
	print(new_fea_name,'done')
all_train['label']=train.label
all_test['label']=test.label
all_train.to_csv(save_path+"train_three_multi_1.csv",index=None)
all_test.to_csv(save_path+"test_three_multi_1.csv",index=None)

t2=datetime.now()
print(t2-t1)
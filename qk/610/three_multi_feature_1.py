import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm
from datetime import datetime

t1=datetime.now()#23个
all_cross=['creativeSize_kw2_topic2_multi_stas', 'aid_kw2_age_multi_stas', 'creativeSize_topic2_LBS_multi_stas', 'creativeSize_kw2_LBS_multi_stas', 'aid_kw2_topic2_multi_stas', 'aid_age_education_multi_stas', 'creativeSize_age_LBS_multi_stas', 'creativeSize_LBS_education_multi_stas', 'aid_kw2_education_multi_stas', 'aid_age_LBS_multi_stas', 'creativeSize_kw2_age_multi_stas', 'aid_topic2_age_multi_stas', 'aid_topic2_education_multi_stas', 'creativeSize_topic2_age_multi_stas', 'aid_LBS_education_multi_stas', 'aid_topic2_interest1_multi_stas', 'aid_interest1_age_multi_stas', 'aid_kw2_LBS_multi_stas', 'aid_kw2_interest1_multi_stas', 'creativeSize_kw2_education_multi_stas', 'creativeSize_topic2_education_multi_stas', 'aid_interest1_LBS_multi_stas', 'creativeSize_age_education_multi_stas']
usecols=['creativeSize', 'kw2', 'topic2', 'aid', 'label']
train=pd.read_csv("train.csv")
train[train.label==-1,'label']=0
test1=pd.read_csv("test1.csv")
test1['label']=-1
test2=pd.read_csv("test2.csv")
test2['label']=-1
test=pd.concat([test1,test2])
adFeature=pd.read_csv("adFeature.csv",usecols=['aid','creativeSize'])
userFeature=pd.read_csv("userFeature.csv",usecols=['uid','kw2','topic2'])
#train_test=pd.concat([train,test1,test2],axis=1)
#train_test=pd.read_csv("../data/train_test1_2_data.csv",usecols=usecols)
train=pd.merge(train,adFeature,on='aid',how='left')
train=pd.merge(train,userFeature,on='uid',how='left')
train=train.fillna('-1')
test=pd.merge(test,adFeature,on='aid',how='left')
test=pd.merge(test,userFeature,on='uid',how='left')
test=test.fillna('-1')

#gc.collect()

def hebing(x,y,z):
	cross=""
	x=str(x)
	y=str(y).split()
	z=str(z).split()
	for i in y:
		for j in z:
			cross+=x+"_"+i+"_"+j+" "
	return cross.strip()


#for i in tqdm(all_cross[0:11]):
# all_train=pd.DataFrame()
# all_test=pd.DataFrame()
# i=all_cross[0]
# x_split=i.split('_')
# a=x_split[0]
# b=x_split[1]
# c=x_split[2]
# new_fea_name=str(a)+'_'+str(b)+'_'+str(c)+'_stats'
# all_train[new_fea_name]=train[[a,b,c]].apply(lambda x:hebing(x[a],x[b],x[c]),axis=1)
# all_test[new_fea_name]=test[[a,b,c]].apply(lambda x:hebing(x[a],x[b],x[c]),axis=1)
# print(new_fea_name,'done')
# all_train['label']=train.label
# all_test['label']=test.label
# all_train.to_csv("{}_train.csv".format(i),index=None)
# all_test.to_csv("{}_test.csv".format(i),index=None)
all_train=pd.DataFrame()
all_test=pd.DataFrame()
i=all_cross[4]
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
all_train.to_csv("{}_train.csv".format(i),index=None)
all_test.to_csv("{}_test.csv".format(i),index=None)




t2=datetime.now()
print(t2-t1)
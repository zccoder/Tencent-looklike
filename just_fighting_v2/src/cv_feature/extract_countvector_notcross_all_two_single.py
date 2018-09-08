import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
from datetime import datetime
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import gc

t1=datetime.now()
print("reading..")
input_path='../../data/'
save_path='../../all_feature/countvector_feature_file/'
data=pd.read_csv(input_path+'train_test_data.csv')
data=data.fillna('-1')
del_vec=['appIdAction','appIdInstall','interest3','interest4','kw3','topic3','creativeSize','aid','advertiserId','campaignId','creativeId','adCategoryId','productId','productType','LBS','age',
'carrier','consumptionAbility','ct','education','gender','house','interest1']
for i in del_vec:
	del data[i]
gc.collect()
#25个外加一个creativeSize
all_vector=['interest2','interest5','kw1','kw2','os','marriageStatus','topic1','topic2']
print(len(all_vector))
train=data[data.label!=-1]
test=data[data.label==-1]
del data
gc.collect()

#train_x=train[['creativeSize']]
#test_x=test[['creativeSize']]
train_x=None
test_x=None

cv=CountVectorizer(token_pattern='\w+',max_features=1000,min_df=10,binary=True)
for feature in tqdm(all_vector):
	train_a=cv.fit_transform(train[feature].astype('str'))
	test_a=cv.transform(test[feature].astype('str'))
	sparse.save_npz("{0}train_x_notcross_{1}".format(save_path,feature),train_a.astype('int8').tocsr())
	sparse.save_npz("{0}test_x_notcross_{1}".format(save_path,feature),test_a.astype('int8').tocsr())
	print(feature," done")
	del train[feature],test[feature]
	gc.collect()

t2=datetime.now()
print(t2-t1)
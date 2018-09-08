import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
from datetime import datetime
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import gc

t1=datetime.now()
print("reading..")
input_path='../../all_feature/uid_pos_neg_file/'
save_path='../../all_feature/countvector_feature_file/'
train=pd.read_csv(input_path+"train_neg_pos_adCategoryId.csv")
test=pd.read_csv(input_path+"test_neg_pos_adCategoryId.csv")
all_vector=['adCategoryId_neg','adCategoryId_pos']
print(len(all_vector))


train_x=None
test_x=None

cv=TfidfVectorizer(token_pattern='\w+')
for feature in tqdm(all_vector):
	cv.fit(train[feature].astype('str'))
	train_a=cv.transform(train[feature].astype('str'))
	test_a=cv.transform(test[feature].astype('str'))
	sparse.save_npz("{0}train_x_notcross_tfidf_{1}".format(save_path,feature),train_a.astype('int8').tocsr())
	sparse.save_npz("{0}test_x_notcross_tfidf_{1}".format(save_path,feature),test_a.astype('int8').tocsr())
	print(feature," done")
	del train[feature],test[feature]
	gc.collect()

t2=datetime.now()
print(t2-t1)
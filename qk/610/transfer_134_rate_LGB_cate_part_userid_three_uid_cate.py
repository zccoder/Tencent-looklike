import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import numpy as np
import gc
from sklearn.externals import joblib
from datetime import datetime
t1=datetime.now()
usecol=['uid','aid','age','gender','education','consumptionAbility','LBS','carrier','house','advertiserId','campaignId','adCategoryId','creativeSize','creativeId','productType','productId']
primary_train=pd.read_csv("../data/train_primary_labelencoder_x.csv",usecols=usecol)
one_hot_train=pd.read_csv("one_hot_statis_feature_train.csv",dtype='float32')#15
vector_len_train=pd.read_csv("vector_len_feature_train.csv",dtype='float32')#5
vector_statis_train=pd.read_csv("vector_statis_feature_train.csv",dtype='float32')#16
three_multi_statis_train_1_2=pd.read_csv("three_multi_statis_feature_train_1_2.csv",dtype='float32')#5
two_single_statis_train_1=pd.read_csv("two_single_statis_feature_train_1.csv",dtype='float32')
two_single_statis_train_2=pd.read_csv("two_single_statis_feature_train_2.csv",dtype='float32')
two_multi_statis_train_1=pd.read_csv("two_multi_statis_feature_train_1.csv",dtype='float32')
two_multi_statis_train_2=pd.read_csv("two_multi_statis_feature_train_2.csv",dtype='float32')
two_multi_statis_train_3=pd.read_csv("two_multi_statis_feature_train_3.csv",dtype='float32')
three_multi_statis_train_1_1_1=pd.read_csv("three_multi_statis_feature_train_1_1_1.csv",dtype='float32')
three_multi_statis_train_1_1_2=pd.read_csv("three_multi_statis_feature_train_1_1_2.csv",dtype='float32')
three_multi_statis_train_1_1_3=pd.read_csv("three_multi_statis_feature_train_1_1_3.csv",dtype='float32')
three_multi_statis_train_1_1_4=pd.read_csv("three_multi_statis_feature_train_1_1_4.csv",dtype='float32')
user_train=pd.read_csv("user_statis_feature_train.csv",dtype='float32',na_values='NULL')
two_single_statis_train_3_1=pd.read_csv("two_single_statis_feature_train_3_1.csv",dtype='float32')
two_single_statis_train_3_2=pd.read_csv("two_single_statis_feature_train_3_2.csv",dtype='float32')
two_single_statis_train_3_3=pd.read_csv("two_single_statis_feature_train_3_3.csv",dtype='float32')
#train_x=np.hstack((primary_train,one_hot_train,vector_len_train,vector_statis_train,three_multi_statis_train_1_2,two_single_statis_train_1,two_single_statis_train_2,two_multi_statis_train_1,two_multi_statis_train_2,two_multi_statis_train_3,three_multi_statis_train_1_1_1,three_multi_statis_train_1_1_2,three_multi_statis_train_1_1_3,three_multi_statis_train_1_1_4,user_train,two_single_statis_train_3_1,two_single_statis_train_3_2,two_single_statis_train_3_3)).astype('float32')
train_y=pd.read_csv('../cv_feature/all_train_y.csv')
train_x, evals_x, train_y, evals_y = train_test_split(train_x,train_y,test_size=0.05, random_state=2018)#训练集和验证集划分
del primary_train,one_hot_train,vector_len_train,vector_statis_train,three_multi_statis_train_1_2,two_single_statis_train_1,two_single_statis_train_2,two_multi_statis_train_1,two_multi_statis_train_2,two_multi_statis_train_3,three_multi_statis_train_1_1_1,three_multi_statis_train_1_1_2,three_multi_statis_train_1_1_3,three_multi_statis_train_1_1_4,user_train,two_single_statis_train_3_1,two_single_statis_train_3_2,two_single_statis_train_3_3
gc.collect()
primary_test=pd.read_csv("../data/test_primary_labelencoder_x.csv",usecols=usecol)
one_hot_test=pd.read_csv("one_hot_statis_feature_test.csv",dtype='float32')#15
vector_len_test=pd.read_csv("vector_len_feature_test.csv",dtype='float32')#5
vector_statis_test=pd.read_csv("vector_statis_feature_test.csv",dtype='float32')#16
three_multi_statis_test_1_2=pd.read_csv("three_multi_statis_feature_test_1_2.csv",dtype='float32')#5
two_single_statis_test_1=pd.read_csv("two_single_statis_feature_test_1.csv",dtype='float32')
two_single_statis_test_2=pd.read_csv("two_single_statis_feature_test_2.csv",dtype='float32')
two_multi_statis_test_1=pd.read_csv("two_multi_statis_feature_test_1.csv",dtype='float32')
two_multi_statis_test_2=pd.read_csv("two_multi_statis_feature_test_2.csv",dtype='float32')
two_multi_statis_test_3=pd.read_csv("two_multi_statis_feature_test_3.csv",dtype='float32')
three_multi_statis_test_1_1_1=pd.read_csv("three_multi_statis_feature_test_1_1_1.csv",dtype='float32')
three_multi_statis_test_1_1_2=pd.read_csv("three_multi_statis_feature_test_1_1_2.csv",dtype='float32')
three_multi_statis_test_1_1_3=pd.read_csv("three_multi_statis_feature_test_1_1_3.csv",dtype='float32')
three_multi_statis_test_1_1_4=pd.read_csv("three_multi_statis_feature_test_1_1_4.csv",dtype='float32')
user_test=pd.read_csv("user_statis_feature_test.csv",dtype='float32',na_values='NULL')
two_single_statis_test_3_1=pd.read_csv("two_single_statis_feature_test_3_1.csv",dtype='float32')
two_single_statis_test_3_2=pd.read_csv("two_single_statis_feature_test_3_2.csv",dtype='float32')
two_single_statis_test_3_3=pd.read_csv("two_single_statis_feature_test_3_3.csv",dtype='float32')
#test_x=np.hstack((primary_test,one_hot_test,vector_len_test,vector_statis_test,three_multi_statis_test_1_2,two_single_statis_test_1,two_single_statis_test_2,two_multi_statis_test_1,two_multi_statis_test_2,two_multi_statis_test_3)).astype('float32')
test_x=np.hstack((primary_test,one_hot_test,vector_len_test,vector_statis_test,three_multi_statis_test_1_2,two_single_statis_test_1,two_single_statis_test_2,two_multi_statis_test_1,two_multi_statis_test_2,two_multi_statis_test_3,three_multi_statis_test_1_1_1,three_multi_statis_test_1_1_2,three_multi_statis_test_1_1_3,three_multi_statis_test_1_1_4,user_test,two_single_statis_test_3_1,two_single_statis_test_3_2,two_single_statis_test_3_3)).astype('float32')
del primary_test,one_hot_test,vector_len_test,vector_statis_test,three_multi_statis_test_1_2,two_single_statis_test_1,two_single_statis_test_2,two_multi_statis_test_1,two_multi_statis_test_2,two_multi_statis_test_3,three_multi_statis_test_1_1_1,three_multi_statis_test_1_1_2,three_multi_statis_test_1_1_3,three_multi_statis_test_1_1_4,user_test,two_single_statis_test_3_1,two_single_statis_test_3_2,two_single_statis_test_3_3
print(train_x.shape)
print(test_x.shape)
print("LGB test")
clf = lgb.LGBMClassifier(
	boosting='gbdt',
        application='binary',
        metric='auc',
        bagging_fraction=0.7,
        feature_fraction=0.7,
        num_leaves=127,
        learning_rate=0.05,
        # 'max_depth': 5,
        max_bin=255,
        min_data_in_leaf=200,
        lambda_l1= 3.0,
        lambda_l2=3.0,
        num_threads=-1,
        seed=2018,
        n_estimators=10000
)
#cate_range="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
cate_range=[i for i in range(16)]
clf.fit(train_x, train_y, eval_set=[(train_x, train_y),(evals_x,evals_y)], categorical_feature=cate_range,eval_metric='auc',early_stopping_rounds=200)
res=pd.read_csv('../data/test12.csv')
res['score'] = clf.predict_proba(test_x)[:,1]
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv('./submission_baseline134_all_1w_cate_part_user_three_uid_cate.csv', index=False)
os.system('zip ./baseline134_all_1w_cate_part_user_three_uid_cate.zip ./submission_baseline134_all_1w_cate_part_user_three_uid_cate.csv')
try:
	joblib.dump(clf,"qk_baseline_newfeature_offline_134_all_cate_part_user_three_uid_cate.pkl")
	print("model have saved")
except Exception as e:
	print(e)
t2=datetime.now()
print("time:",t2-t1)
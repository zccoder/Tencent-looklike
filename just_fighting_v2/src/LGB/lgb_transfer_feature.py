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
input_path='../../data/'
transfer_path='../../all_feature/transfer_feature_file/'
model_path='../../models/'
save_path='../../submission_result/'
train_y=pd.read_csv(input_path+'all_train_y.csv')
res=pd.read_csv(input_path+'test.csv')
usecol=['uid','aid','age','gender','education','consumptionAbility','LBS','carrier','house','advertiserId','campaignId','adCategoryId','creativeSize','creativeId','productType','productId']

primary_train=pd.read_csv(input_path+"train_primary_labelencoder_x.csv",usecols=usecol)
one_hot_train=pd.read_csv(transfer_path+"one_hot_statis_feature_train.csv",dtype='float32')#15
vector_len_train=pd.read_csv(transfer_path+"vector_len_feature_train.csv",dtype='float32')#5
vector_statis_train=pd.read_csv(transfer_path+"vector_statis_feature_train.csv",dtype='float32')#16
three_multi_statis_train_1_2=pd.read_csv(transfer_path+"three_multi_statis_feature_train_1_2.csv",dtype='float32')#5
two_single_statis_train_1=pd.read_csv(transfer_path+"two_single_statis_feature_train_1.csv",dtype='float32')
two_single_statis_train_2=pd.read_csv(transfer_path+"two_single_statis_feature_train_2.csv",dtype='float32')
two_multi_statis_train_1=pd.read_csv(transfer_path+"two_multi_statis_feature_train_1.csv",dtype='float32')
two_multi_statis_train_2=pd.read_csv(transfer_path+"two_multi_statis_feature_train_2.csv",dtype='float32')
two_multi_statis_train_3=pd.read_csv(transfer_path+"two_multi_statis_feature_train_3.csv",dtype='float32')
three_multi_statis_train_1_1_1=pd.read_csv(transfer_path+"three_multi_statis_feature_train_1_1_1.csv",dtype='float32')
three_multi_statis_train_1_1_2=pd.read_csv(transfer_path+"three_multi_statis_feature_train_1_1_2.csv",dtype='float32')
three_multi_statis_train_1_1_3=pd.read_csv(transfer_path+"three_multi_statis_feature_train_1_1_3.csv",dtype='float32')
three_multi_statis_train_1_1_4=pd.read_csv(transfer_path+"three_multi_statis_feature_train_1_1_4.csv",dtype='float32')
three_multi_statis_train_1_1_5=pd.read_csv(transfer_path+"three_multi_statis_feature_train_1_1_5.csv",dtype='float32')
three_multi_sattis_train_2_1=pd.read_csv(transfer_path+"three_multi_statis_feature_train_2_1.csv",dtype='float32')
three_multi_sattis_train_2_2=pd.read_csv(transfer_path+"three_multi_statis_feature_train_2_2.csv",dtype='float32')
user_train=pd.read_csv(transfer_path+"user_statis_feature_train.csv",dtype='float32',na_values='NULL')
two_single_statis_train_3_1=pd.read_csv(transfer_path+"two_single_statis_feature_train_3_1.csv",dtype='float32')
two_single_statis_train_3_2=pd.read_csv(transfer_path+"two_single_statis_feature_train_3_2.csv",dtype='float32')
two_single_statis_train_3_3=pd.read_csv(transfer_path+"two_single_statis_feature_train_3_3.csv",dtype='float32')
train_x=np.hstack((primary_train,one_hot_train,vector_len_train,vector_statis_train,three_multi_statis_train_1_2,two_single_statis_train_1,two_single_statis_train_2,two_multi_statis_train_1,two_multi_statis_train_2,two_multi_statis_train_3,three_multi_statis_train_1_1_1,three_multi_statis_train_1_1_2,three_multi_statis_train_1_1_3,three_multi_statis_train_1_1_4,three_multi_statis_train_1_1_5,three_multi_sattis_train_2_1,three_multi_sattis_train_2_2,user_train,two_single_statis_train_3_1,two_single_statis_train_3_2,two_single_statis_train_3_3)).astype('float32')

train_x, evals_x, train_y, evals_y = train_test_split(train_x,train_y,test_size=0.05, random_state=2018)#训练集和验证集划分
del primary_train,one_hot_train,vector_len_train,vector_statis_train,three_multi_statis_train_1_2,two_single_statis_train_1,two_single_statis_train_2,two_multi_statis_train_1,two_multi_statis_train_2,two_multi_statis_train_3,three_multi_statis_train_1_1_1,three_multi_statis_train_1_1_2,three_multi_statis_train_1_1_3,three_multi_statis_train_1_1_4,three_multi_statis_train_1_1_5,three_multi_sattis_train_2_1,three_multi_sattis_train_2_2,user_train,two_single_statis_train_3_1,two_single_statis_train_3_2,two_single_statis_train_3_3
gc.collect()
primary_test=pd.read_csv(input_path+"test_primary_labelencoder_x.csv",usecols=usecol)
one_hot_test=pd.read_csv(transfer_path+"one_hot_statis_feature_test.csv",dtype='float32')#15
vector_len_test=pd.read_csv(transfer_path+"vector_len_feature_test.csv",dtype='float32')#5
vector_statis_test=pd.read_csv(transfer_path+"vector_statis_feature_test.csv",dtype='float32')#16
three_multi_statis_test_1_2=pd.read_csv(transfer_path+"three_multi_statis_feature_test_1_2.csv",dtype='float32')#5
two_single_statis_test_1=pd.read_csv(transfer_path+"two_single_statis_feature_test_1.csv",dtype='float32')
two_single_statis_test_2=pd.read_csv(transfer_path+"two_single_statis_feature_test_2.csv",dtype='float32')
two_multi_statis_test_1=pd.read_csv(transfer_path+"two_multi_statis_feature_test_1.csv",dtype='float32')
two_multi_statis_test_2=pd.read_csv(transfer_path+"two_multi_statis_feature_test_2.csv",dtype='float32')
two_multi_statis_test_3=pd.read_csv(transfer_path+"two_multi_statis_feature_test_3.csv",dtype='float32')
three_multi_statis_test_1_1_1=pd.read_csv(transfer_path+"three_multi_statis_feature_test_1_1_1.csv",dtype='float32')
three_multi_statis_test_1_1_2=pd.read_csv(transfer_path+"three_multi_statis_feature_test_1_1_2.csv",dtype='float32')
three_multi_statis_test_1_1_3=pd.read_csv(transfer_path+"three_multi_statis_feature_test_1_1_3.csv",dtype='float32')
three_multi_statis_test_1_1_4=pd.read_csv(transfer_path+"three_multi_statis_feature_test_1_1_4.csv",dtype='float32')
three_multi_statis_test_1_1_5=pd.read_csv(transfer_path+"three_multi_statis_feature_test_1_1_5.csv",dtype='float32')
three_multi_statis_test_2_1=pd.read_csv(transfer_path+"three_multi_statis_feature_test_2_1.csv",dtype='float32')
three_multi_statis_test_2_2=pd.read_csv(transfer_path+"three_multi_statis_feature_test_2_2.csv",dtype='float32')
user_test=pd.read_csv(transfer_path+"user_statis_feature_test.csv",dtype='float32',na_values='NULL')
two_single_statis_test_3_1=pd.read_csv(transfer_path+"two_single_statis_feature_test_3_1.csv",dtype='float32')
two_single_statis_test_3_2=pd.read_csv(transfer_path+"two_single_statis_feature_test_3_2.csv",dtype='float32')
two_single_statis_test_3_3=pd.read_csv(transfer_path+"two_single_statis_feature_test_3_3.csv",dtype='float32')
test_x=np.hstack((primary_test,one_hot_test,vector_len_test,vector_statis_test,three_multi_statis_test_1_2,two_single_statis_test_1,two_single_statis_test_2,two_multi_statis_test_1,two_multi_statis_test_2,two_multi_statis_test_3,three_multi_statis_test_1_1_1,three_multi_statis_test_1_1_2,three_multi_statis_test_1_1_3,three_multi_statis_test_1_1_4,three_multi_statis_test_1_1_5,three_multi_statis_test_2_1,three_multi_statis_test_2_2,user_test,two_single_statis_test_3_1,two_single_statis_test_3_2,two_single_statis_test_3_3)).astype('float32')
del primary_test,one_hot_test,vector_len_test,vector_statis_test,three_multi_statis_test_1_2,two_single_statis_test_1,two_single_statis_test_2,two_multi_statis_test_1,two_multi_statis_test_2,two_multi_statis_test_3,three_multi_statis_test_1_1_1,three_multi_statis_test_1_1_2,three_multi_statis_test_1_1_3,three_multi_statis_test_1_1_4,three_multi_statis_test_1_1_5,three_multi_statis_test_2_1,three_multi_statis_test_2_2,user_test,two_single_statis_test_3_1,two_single_statis_test_3_2,two_single_statis_test_3_3
print(train_x.shape)
print(test_x.shape)
print("LGB test")
clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        metric='auc',
        subsample=0.7,
        colsample_bytree=0.7,
        num_leaves=127,
        learning_rate=0.05,
        max_bin=255,
        min_child_samples=200,
        reg_alpha= 3.0,
        reg_lambda=3.0,
        nthread=-1,
        seed=2018,
        n_estimators=5000
)

clf.fit(train_x, train_y, eval_set=[(train_x, train_y),(evals_x,evals_y)], eval_metric='auc',early_stopping_rounds=200)
res['score'] = clf.predict_proba(test_x)[:,1]
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv(save_path+'submission_LGB.csv', index=False)
try:
	joblib.dump(clf,model_path+"LGB_model.pkl")
	print("LGB model have saved")
except Exception as e:
	print(e)
t2=datetime.now()
print("time:",t2-t1)
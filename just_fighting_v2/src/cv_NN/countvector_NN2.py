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
import os; 
#os.environ['OMP_NUM_THREADS'] = '4' 
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict
import tensorflow as tf
import keras as ks
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error,mean_squared_error
from sklearn.model_selection import KFold
import math
import collections
from keras.utils import to_categorical 
from keras import backend as K
import itertools
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
from scipy.sparse import csr_matrix
from sklearn import preprocessing
t1=datetime.now()
@contextmanager  #创建一个上下文管理器，显示运行的时间
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
def qk_auc(y_true, y_pred):  
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)  
    binSizes = -(pfas[1:]-pfas[:-1])  
    s = ptas*binSizes  
    return K.sum(s, axis=0)  
#-----------------------------------------------------------------------------------------------------------------------------------------------------  
# PFA, prob false alert for binary classifier  
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):  
    y_pred = K.cast(y_pred >= threshold, 'float32')  
    # N = total number of negative labels  
    N = K.sum(1 - y_true)  
    # FP = total number of false alerts, alerts from the negative class labels  
    FP = K.sum(y_pred - y_pred * y_true)  
    return FP/N  
#-----------------------------------------------------------------------------------------------------------------------------------------------------  
# P_TA prob true alerts for binary classifier  
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):  
    y_pred = K.cast(y_pred >= threshold, 'float32')  
    # P = total number of positive labels  
    P = K.sum(y_true)  
    # TP = total number of correct alerts, alerts from the positive class labels  
    TP = K.sum(y_pred * y_true)  
    return TP/P  
#def fit_predict(xs, y_train,evals_x,evals_y):
def fit_predict(xs,model_path, y_train):
    all_split_train_x,all_test_x,index=xs
    with tf.Session(graph=tf.Graph()) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        all_model_input_list=[]
        for model_input in all_split_train_x:
        	all_model_input_list.append(ks.Input(shape=(model_input.shape[1],),sparse=True))        
        deep=concatenate(all_model_input_list)
        deep=ks.layers.Dense(1536,activation='relu')(deep)
        deep = ks.layers.Dense(768, activation='relu')(deep)
        deep = ks.layers.Dense(768, activation='relu')(deep)
        out = ks.layers.Dense(1,activation='sigmoid')(deep)
        model = ks.Model(all_model_input_list, out)
        model.compile(loss='binary_crossentropy',optimizer=ks.optimizers.Adam(lr=1e-3,decay=0.0005,amsgrad=True))
        for i in range(2):
            with timer(f'epoch {i + 1}'):
                model.fit(x=all_split_train_x, y=y_train,batch_size=2**(12+i), epochs=1, verbose=2)
        model.save_weights(model_path+"two_model_weigth_{0}.h5".format(index))
        model.save(model_path+"two_model_{0}.h5".format(index))
        pre=model.predict(all_test_x,batch_size=16384)
        return pre

input_path='../../data/'
transfer_path='../../all_feature/transfer_feature_file/'
countvector_path='../../all_feature/countvector_feature_file/'
save_path='../../submission_result/'
model_path='../../models/'
res=pd.read_csv(input_path+"test.csv")
train_y=pd.read_csv(input_path+'all_train_y.csv').label.values

all_vector=['creativeSize','aid','advertiserId','campaignId','creativeId','adCategoryId','productId','productType','LBS','age','appIdAction','appIdInstall',
'carrier','consumptionAbility','ct','education','gender','house','interest1','interest2','interest3',
'interest4','interest5','kw1','kw2','kw3','os','marriageStatus','topic1','topic2','topic3','aid_pos','aid_neg']

tfidf_vector=['productType_pos','productType_neg','adCategoryId_pos','adCategoryId_neg','productId_pos','productId_neg','creativeSize_pos','creativeSize_neg']
train_transfer=pd.read_csv(transfer_path+"three_multi_statis_feature_train_2_1.csv")
train_transfer1=pd.read_csv(transfer_path+"three_multi_statis_feature_train_2_2.csv")
train_transfer2=pd.read_csv(transfer_path+"user_statis_feature_train.csv")
train_transfer=train_transfer.fillna(train_transfer.median())
train_transfer1=train_transfer1.fillna(train_transfer1.median())
train_transfer2=train_transfer2.fillna(train_transfer2.median())
scaler=preprocessing.StandardScaler()
scaler1=preprocessing.StandardScaler()
scaler2=preprocessing.StandardScaler()
train_transfer=scaler.fit_transform(train_transfer)
train_transfer1=scaler1.fit_transform(train_transfer1)
train_transfer2=scaler2.fit_transform(train_transfer2)
train_transfer=csr_matrix(train_transfer)
train_transfer1=csr_matrix(train_transfer1)
train_transfer2=csr_matrix(train_transfer2)
all_train_x=[]
for i in all_vector:
	all_train_x.append(sparse.load_npz("{0}train_x_notcross_{1}.npz".format(countvector_path,i)))
for i in tfidf_vector:
	all_train_x.append(sparse.load_npz("{0}train_x_notcross_tfidf_{1}.npz".format(countvector_path,i)))
all_train_x.append(train_transfer)
all_train_x.append(train_transfer1)
all_train_x.append(train_transfer2)
del train_transfer,train_transfer1,train_transfer2
gc.collect()

test_transfer=pd.read_csv(transfer_path+"three_multi_statis_feature_test_2_1.csv")
test_transfer=test_transfer.fillna(test_transfer.median())
test_transfer1=pd.read_csv(transfer_path+"three_multi_statis_feature_test_2_2.csv")
test_transfer1=test_transfer1.fillna(test_transfer1.median())
test_transfer2=pd.read_csv(transfer_path+"user_statis_feature_test.csv")
test_transfer2=test_transfer2.fillna(test_transfer2.median())
test_transfer=scaler.fit_transform(test_transfer)
test_transfer1=scaler1.fit_transform(test_transfer1)
test_transfer2=scaler2.fit_transform(test_transfer2)
test_transfer=csr_matrix(test_transfer)
test_transfer1=csr_matrix(test_transfer1)
test_transfer2=csr_matrix(test_transfer2)

all_test_x=[]
for i in all_vector:
	all_test_x.append(sparse.load_npz("{0}test_x_notcross_{1}.npz".format(countvector_path,i)))
for i in tfidf_vector:
	all_test_x.append(sparse.load_npz("{0}test_x_notcross_tfidf_{1}.npz".format(countvector_path,i)))
all_test_x.append(test_transfer)
all_test_x.append(test_transfer1)
all_test_x.append(test_transfer2)
del test_transfer,test_transfer1,test_transfer2
gc.collect()

bagging_times=10
xs=[]
for i in range(bagging_times):
	xs.append([all_train_x,all_test_x,i])

with ThreadPool(processes=4) as pool:
	allpre=pool.map(partial(fit_predict,model_path=model_path,y_train=train_y),xs)

pre=np.mean(allpre,axis=0)
res['score'] = pre
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv(save_path+'cv_NN2.csv', index=False)
#os.system('zip ./baseline_NN_pos_neg_three_ams_grad_scler1_four.zip ./submission_baseline_NN_pos_neg_three_ams_grad_scaler1_four.csv')
t2=datetime.now()
print("time:",t2-t1)
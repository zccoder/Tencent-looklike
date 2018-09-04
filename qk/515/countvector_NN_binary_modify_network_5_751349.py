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
def fit_predict(xs, y_train):
    #[ [[Xb_train, Xb_valid], [X_train, X_valid]] ,[[Xb_train, Xb_valid], [X_train, X_valid]] ]
    X_train, X_test = xs   
    print("X_train:",X_train.shape)
    print("X_test:",X_test.shape)
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32',sparse=True)
        #model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(384, activation='relu')(model_in)
        out = ks.layers.Dense(192, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1,activation='sigmoid')(out)
        model = ks.Model(model_in, out)
        print(model.summary())
        model.compile(loss='binary_crossentropy', optimizer=ks.optimizers.Adam(lr=1e-3))
        for i in range(3):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train,batch_size=2**(11 + i), epochs=1, verbose=2)
        test_pre=model.predict(X_test)
        print(test_pre)
        return test_pre

#res=pd.read_csv('res.csv')#需要提交的结果
res=pd.read_csv("test2.csv")

train_x=sparse.load_npz('train_x_notcross_B.npz').astype('float32').tocsr()
test_x=sparse.load_npz('test_x_notcross_B.npz').astype('float32').tocsr()
print(train_x.shape)
print(test_x.shape)

train_y=pd.read_csv('train_y_alldata.csv',header=None)
train_y=np.array(train_y)
#train_x, evals_x, train_y, evals_y = train_test_split(train_x,train_y,test_size=0.05, random_state=2018)#训练集和验证集划分
#train_y=to_categorical(train_y)
kf=KFold(n_splits=5,shuffle=True,random_state=2018)
y_pred_all=[]
for i,(train_index,valid_index) in enumerate(kf.split(train_x)):
	with ThreadPool(processes=20) as pool:
	    train_xx=train_x[train_index]
	    train_yy=train_y[train_index]
	    #evals_xx=train_x[valid_index]
	    #evals_yy=train_y[valid_index]
	    xs=[[train_xx,test_x]]*10
	    allpre=pool.map(partial(fit_predict,y_train=train_yy),xs)
	    y_pred=np.mean(allpre,axis=0)
	    y_pred_all.append(y_pred)

    #pre=fit_predict([train_x,test_x],train_y,evals_x,evals_y)
pre=np.mean(y_pred_all,axis=0)
#clf.fit(train_x, train_y, eval_set=[(train_x, train_y),(evals_x,evals_y)], eval_metric='auc',early_stopping_rounds=500)
res['score'] = pre
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv('./submission_baseline_NN_5_five_fold_big.csv', index=False)
os.system('zip ./baseline_NN_5_five_fold_big.zip ./submission_baseline_NN_5_five_fold_big.csv')


t2=datetime.now()
print("time:",t2-t1)
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

def jacek_auc(y_true, y_pred):
   score, up_opt = tf.metrics.auc(y_true, y_pred)
   #score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

def fit_predict(xs, y_train,evals_y,evals_x):
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
        out = ks.layers.Dense(768, activation='relu')(model_in)
        out = ks.layers.Dense(256, activation='relu')(out)
        out = ks.layers.Dense(256, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1,activation='sigmoid')(out)
        #out= ks.layers.Activation('softmax')(out)
        model = ks.Model(model_in, out)
        print(model.summary())
        model.compile(loss='binary_crossentropy', metrics=[jacek_auc], optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(4):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(12 + i), epochs=1, verbose=1,
                    validation_data=(evals_x,evals_y),shuffle=False)
                # train_pre=model.predict(X_train)
                # print('trainauc:',jacek_auc(y_train,train_pre))
                # evals_pre=model.predict(evals_x)
                # print('testauc:',jacek_auc(evals_y,evals_x))
        test_pre=model.predict(X_test)
        print(test_pre)
        return test_pre

res=pd.read_csv('res.csv')#需要提交的结果

train_x=sparse.load_npz('train_x_notcross.npz').astype('float32').tocsr()
test_x=sparse.load_npz('test_x_notcross.npz').astype('float32').tocsr()
print(train_x.shape)#
print(test_x.shape)


train_y=pd.read_csv('train_y_alldata.csv',header=None)

train_x, evals_x, train_y, evals_y = train_test_split(train_x,train_y,test_size=0.05, random_state=2018)#训练集和验证集划分
#train_y=to_categorical(train_y)
#evals_y=to_categorical(evals_y)

pre=fit_predict([train_x,test_x],train_y,evals_y,evals_x)

#clf.fit(train_x, train_y, eval_set=[(train_x, train_y),(evals_x,evals_y)], eval_metric='auc',early_stopping_rounds=500)
res['score'] = pre
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv('./submission_baseline15_NN_metric.csv', index=False)
os.system('zip ./baseline15_NN_metric.zip ./submission_baseline15_NN_metric.csv')


t2=datetime.now()
print("time:",t2-t1)
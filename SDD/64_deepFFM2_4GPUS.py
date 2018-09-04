import numpy as np
import pandas as pd
import time
import gc
import sys
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
import itertools
from keras import backend  as KK
#from keras.engine.topology import Layer
from keras.metrics import categorical_accuracy
from keras.utils import multi_gpu_model
gpus_num=4
import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

single_emb=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType','creativeSize','new_AID']
singel_max=[898,5,3,2,7,3,1,4,64,27,197,479,831,74,83,3,14]

#sparse_col_name=['appIdAction','appIdInstall','interest1','interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3','topic1', 'topic2', 'topic3']
#sparse_col_max_length=[537,920,47,32,10,10,86,5,5,5,5,5,5]
#max_value=[-1,-1,122,82,10,10,136,796741,121958,58782,9999,9999,9463]
'''
全部数据 68996077
3:  interest1  maxlen 33  maxvalue  122 ok! 
4:  interest2  maxlen 33  maxvalue  82  ok!
5:  interest3  maxlen 10  maxvalue  10  ok!
6:  interest4  maxlen 10  maxvalue  10  ok!   
7:  interest5  maxlen 33  maxvalue  136 ok!   
8:  kw1        maxlen 5   maxvalue  796741  ok! 
9:  kw2        maxlen 5   maxvalue  121958  ok! 
10: kw3        maxlen 5   maxvalue  58782   ok! 
11: topic1     maxlen 5   maxvalue  9999  ok! 
12: topic2     maxlen 5   maxvalue  9999  ok! 
13: topic3     maxlen 5   maxvalue  9463  ok! 
'''
#model_col_name=['interest3','interest4','interest5','kw1', 'kw2','kw3','topic1', 'topic2', 'topic3']
#model_col_max_length=[10,10,33,5,5,5,5,5,5]
#model_col_max_value=[10,10,136,796741,121958,58782,9999,9999,9463]
  
model_col_name=['interest1','interest2','interest3','interest4','interest5','kw1', 'kw2','kw3','topic1', 'topic2', 'topic3']
model_col_max_length=[33,33,10,10,33,5,5,5,5,5,5]
model_col_max_value=[122,82,10,10,136,796741,121958,58782,9999,9999,9463]
  
  
  
def binary_crossentropy_with_ranking(y_true, y_pred):
    """ Trying to combine ranking loss with numeric precision"""
    # first get the log loss like normal
    logloss = KK.mean(KK.binary_crossentropy( y_true,y_pred), axis=-1)
    # next, build a rank loss
    # clip the probabilities to keep stability
    y_pred_clipped = KK.clip(y_pred, KK.epsilon(), 1-KK.epsilon())
    # translate into the raw scores before the logit
    y_pred_score = KK.log(y_pred_clipped / (1 - y_pred_clipped))
    # determine what the maximum score for a zero outcome is
    y_pred_score_zerooutcome_max = KK.max(tf.boolean_mask(y_pred_score ,(y_true < 1)))
    # determine how much each score is above or below it
    rankloss = y_pred_score - y_pred_score_zerooutcome_max
    # only keep losses for positive outcomes
    rankloss = tf.boolean_mask(rankloss,tf.equal(y_true,1))
    # only keep losses where the score is below the max
    rankloss = KK.square(KK.clip(rankloss, -100, 0))
    # average the loss for just the positive outcomes
    #tf.reduce_sum(tf.cast(myOtherTensor, tf.float32))
    rankloss = KK.sum(rankloss, axis=-1) / (KK.sum(KK.cast(y_true > 0,tf.float32) + 1))
    return (rankloss + 1)* logloss #- an alternative to try
    #return logloss

# PFA, prob false alert for binary classifier  
def binary_PFA(y_true, y_pred, threshold=KK.variable(value=0.5)):  
    y_pred = KK.cast(y_pred >= threshold, 'float32')  
    # N = total number of negative labels  
    N = KK.sum(1 - y_true)  
    # FP = total number of false alerts, alerts from the negative class labels  
    FP = KK.sum(y_pred - y_pred * y_true)  
    return FP/N 


# P_TA prob true alerts for binary classifier  
def binary_PTA(y_true, y_pred, threshold=KK.variable(value=0.5)):  
    y_pred = KK.cast(y_pred >= threshold, 'float32')  
    # P = total number of positive labels  
    P = KK.sum(y_true)  
    # TP = total number of correct alerts, alerts from the positive class labels  
    TP = KK.sum(y_pred * y_true)  
    return TP/P

def auc(y_true, y_pred):  
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)  
    binSizes = -(pfas[1:]-pfas[:-1])  
    s = ptas*binSizes  
    return KK.sum(s, axis=0)  


def log_loss(y_true, y_pred):
    """ Trying to combine ranking loss with numeric precision"""
    # first get the log loss like normal
    logloss = KK.sum(KK.binary_crossentropy(y_true,y_pred), axis=-1)
    return logloss

def Mean_layer(x):
    return KK.mean(x,axis=1)

def build_model():
    field_sizes=len(single_emb)+len(model_col_name)
    emb_n=4
    FFM_layers=[]
    inputs = []
    flatten_layers=[]
    columns = range(len(single_emb))
    ###------second order term-------###
    for c in columns:
        inputs_c = Input(shape=(1,), dtype='int32',name = 'input_%s'%single_emb[c])
        num_c = singel_max[c]+1
        inputs.append(inputs_c)
        print (num_c,c)
        embed_c = Embedding(
                        num_c,
                        emb_n,
                        input_length=1,
                        name = 'embed_%s'%single_emb[c]
                        #W_regularizer=l2_reg(l2_fm)
                        )(inputs_c)
       
        flatten_c = Reshape((emb_n,))(embed_c)
        flatten_layers.append(flatten_c)

        FFM_temp=[Embedding(num_c,emb_n,input_length=1)(inputs_c) for i_i in range(field_sizes)]
        FFM_layers.append(FFM_temp)
    #field

    for f in range(len(model_col_name)):
        inputs_f = Input(shape=(model_col_max_length[f],),name = 'input_%s'%model_col_name[f])
        num_f = model_col_max_value[f]+1
        inputs.append(inputs_f)
        #print (num_f,f)
        embed_f = Embedding(
                        num_f,
                        emb_n,
                        input_length=model_col_max_length[f],
                        name = 'embed_%s'%model_col_name[f]
                        #W_regularizer=l2_reg(l2_fm)
                        )(inputs_f)
        embed_f=Lambda(Mean_layer)(embed_f)
        flatten_f = Reshape((emb_n,))(embed_f)
        flatten_layers.append(flatten_f)        
		
        FFM_temp=[Lambda(Mean_layer)(Embedding(num_f,emb_n,input_length=model_col_max_length[f])(inputs_c)) for i_i in range(field_sizes)]
        FFM_layers.append(FFM_temp)		
		
	#FFM
    FFM_product=[]
    for ff_i in range(field_sizes):
        for ff_j in range(ff_i+1,field_sizes):
           FFM_product.append(Reshape((emb_n,))(multiply([FFM_layers[ff_i][ff_j],FFM_layers[ff_j][ff_i]])))
    #FFM_second_order=add(FFM_product)
    FFM_second_order=concatenate(FFM_product)

    #second layer
   # summed_features_emb = add(flatten_layers)            ####  None * K
   # summed_features_emb_square = multiply([summed_features_emb,summed_features_emb]) ##### None * K
   # squared_features_emb = []
   # for layer in flatten_layers:
  #       squared_features_emb.append(multiply([layer,layer]))
   # squared_sum_features_emb = add(squared_features_emb)                             ###### None * K
   # subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],output_shape=lambda shapes: shapes[0])
   # y_second_order = subtract_layer([summed_features_emb_square, squared_sum_features_emb])
   # y_second_order  = Lambda(lambda x: x * 0.5)(y_second_order)
   # y_second_order = Dropout(0.8)(y_second_order)
    #___________________________________________________________--
    fm_layers=[]
    for em1,em2 in itertools.combinations(flatten_layers,2):
       dot_layer=merge([em1,em2],mode='dot',dot_axes=1)
       fm_layers.append(dot_layer)
    fm_layers=concatenate(fm_layers)
    #y_first_order = add(flatten_layers) 
    #y_first_order = BatchNormalization()(y_first_order)
    #y_first_order = Dropout(0.8)(y_first_order)
    #------------------------------------------------------------
    y_deep = concatenate(flatten_layers)  
    #y_deep = Dense(256)(y_deep)  #加入BN会变差  PReLU 优于Relu 0.5 is 7483
    #y_deep = Activation('relu',name='output_1')(y_deep)
    #y_deep = Dropout(0.5)(y_deep)
    #y_deep=  Dense(128)(y_deep)
    #y_deep = Activation('relu',name='output_2')(y_deep)
    #y_deep = Dropout(0.5)(y_deep)

	
    #concat_input = concatenate([fm_layers,y_first_order,y_deep],axis=1)
    y_deep = Dense(256)(y_deep)  #加入BN会变差  PReLU 优于Relu 0.5 is 7483
    y_deep = Activation('relu',name='output_3')(y_deep)
    y_deep = Dropout(0.5)(y_deep)
    y_deep=  Dense(128)(y_deep)
    y_deep = Activation('relu',name='output_4')(y_deep)
    y_deep = Dropout(0.5)(y_deep)	
	
    new_input = concatenate([fm_layers,y_deep,FFM_second_order],axis=1)
    outp = Dense(1,activation='sigmoid')(new_input)

    model = Model(inputs=inputs, outputs=outp,name='model')
    optimizer_adam = Adam(lr=0.002)
    parallel_model = keras.utils.training_utils.multi_gpu_model(model,gpus=gpus_num)
    parallel_model.compile(optimizer=optimizer_adam,loss='binary_crossentropy',metrics=[auc,log_loss])
	
	
    #if(loss_flag==0):
    #  model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=[auc,log_loss])
    #elif(loss_flag==1):
    #  model.compile(loss=binary_crossentropy_with_ranking,optimizer=optimizer_adam,metrics=[auc,log_loss])
   # model.summary()
    return parallel_model

#read data
#------------------------------------------------------------------------------
t1=time.time()
single_onehot=pd.read_csv("./SDD_data/final_sdd_single_onehot_embedding_feature2.csv",dtype='float32')
single_onehot['new_AID']=LabelEncoder().fit_transform(single_onehot['aid'].apply(int))
print("read single_onehot over",time.time()-t1)
tem_max=single_onehot['new_AID'].max()
print("NEWID max",tem_max)
singel_max.append(tem_max)
#-**************************************************************************************
train_set=single_onehot[single_onehot['label']!=-1].values[:,3:] #train data
label_train=single_onehot[single_onehot['label']!=-1].values[:,2:3] #label data
test_set=single_onehot[single_onehot['label']==-1].values[:,3:] #test data
subfile=single_onehot[single_onehot['label']==-1][['aid','uid','label']]
del single_onehot;gc.collect()
print("seg over",time.time()-t1)
#-**************************************************************************************
ramdom_seed=0
spilt_prob=0.05
train_x, evals_x, train_y, evals_y=train_test_split(train_set,label_train,test_size=spilt_prob, random_state=ramdom_seed)
del train_set;gc.collect()
print('split data over!',time.time()-t1)
#-**************************************************************************************

X_train=[]
X_valid=[]
X_test=[]
for i in range(len(single_emb)-1):
   X_train.append(train_x[:,i:i+1])
   X_valid.append(evals_x[:,i:i+1])
   X_test.append(test_set[:,i:i+1])
X_train.append(train_x[:,(len(single_emb)-1):])
X_valid.append(evals_x[:,(len(single_emb)-1):])
X_test.append(test_set[:,(len(single_emb)-1):])##
print('input data over!',time.time()-t1)
#-**************************************************************************************
for i in range(len(model_col_name)):
   temp_file=pd.read_csv("./SDD_data/final_sdd_embedding_feature_mix_chusai_%s.csv"%model_col_name[i],dtype='float32')
   print("read %s over"%model_col_name[i],time.time()-t1)
   temp_train=temp_file[temp_file['label']!=-1].values[:,3:] #train data
   temp_test=temp_file[temp_file['label']==-1].values[:,3:] #test data
   del temp_file;gc.collect()
   temp_train_x, temp_evals_x, temp_train_y, temp_evals_y=train_test_split(temp_train,label_train,test_size=spilt_prob, random_state=ramdom_seed)
   del temp_train;gc.collect()
   X_train.append(temp_train_x)
   X_valid.append(temp_evals_x)
   X_test.append(temp_test)##
   del temp_train_x,temp_evals_x;gc.collect()
#-**************************************************************************************



model=build_model()

batch_size=4096
model.fit(X_train,train_y, batch_size=batch_size, validation_data=(X_valid,evals_y),epochs=1, shuffle=True)
print('fit model over!',time.time()-t1)

y_pred_d = model.predict(X_valid,batch_size=4096)  
print('predict over!',time.time()-t1)
from sklearn.metrics import roc_auc_score,log_loss
print ('AUC:',roc_auc_score(evals_y,y_pred_d))
print ('log_loss:',log_loss(evals_y,y_pred_d))
print('compute AUC over!',time.time()-t1)

#pre1=model.predict(X_test,batch_size=10000) 
#subfile['label']=pre1
#subfile.to_csv("./sdd_results/61_1w_4ceng_14.csv",header=True,index=False)
#print('generate result1 is over:',time.time()-t1)


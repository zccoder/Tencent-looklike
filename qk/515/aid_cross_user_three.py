import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import gc
import os
from datetime import datetime
import collections
import itertools

t1=datetime.now()
ad_feature=pd.read_csv("adFeature.csv")

user_feature=pd.read_csv("userFeature.csv")

train=pd.read_csv("train.csv")
predict=pd.read_csv("test1.csv")

train=pd.merge(train,user_feature,on='uid',how='left')
train=pd.merge(train,ad_feature,on='aid',how='left')
predict=pd.merge(predict,user_feature,on='uid',how='left')
predict=pd.merge(predict,ad_feature,on='aid',how='left')
train.loc[train['label']==-1,'label']=0
train=train.fillna('-1')
predict=predict.fillna('-1')
print("reading finish")

def func(x_key_list,fea_count_dict,fea_click_count_dict): ##返回点击的平均值，浏览的平均值，转换率平均值
    x_key_list=str(x_key_list).split()
    count_times=0.0001
    click_count_times=0.0001
    click_rate=0.0
    for i in x_key_list:
        count_times+=fea_count_dict[str(i)]
        click_count_times+=fea_click_count_dict[str(i)]
    if click_count_times==0.0001:
        click_rate=0
    else:
        click_rate=100*float(click_count_times/count_times/len(x_key_list))
        click_rate=np.round(click_rate,4)
    return click_rate

def get_fea_click_count_dict(train,feat):
    fea_count=train[feat].apply(lambda x:x.split(' ')).values
    fea_click_count=train[train.label==1][feat].apply(lambda x:x.split(' ')).values

    fea_count_out = list(itertools.chain.from_iterable(fea_count))
    fea_click_count_out=list(itertools.chain.from_iterable(fea_click_count))

    fea_count_dict=collections.Counter(fea_count_out)
    fea_click_count_dict=collections.Counter(fea_click_count_out)
    return fea_count_dict,fea_click_count_dict

def Feature(train,predict,feat):
    train['index']=list(range(train.shape[0]))
    predict['index']=list(range(predict.shape[0]))
    df_stas_feat=None
    kf=KFold(n_splits=5,random_state=2018,shuffle=True)
    for train_index,val_index in kf.split(train):
        X_train=train.loc[train_index,:]
        X_val=train.loc[val_index,:]
        fea_count_dict,fea_click_count_dict=get_fea_click_count_dict(X_train,feat)   

        X_val[feat+'_stas']=X_val[feat].apply(lambda x:func(x,fea_count_dict,fea_click_count_dict))
        df_stas_feat=pd.concat([df_stas_feat,X_val[[feat+'_stas','index']]],axis=0)
    train=pd.merge(train,df_stas_feat,how='left',on='index')
    fea_count_dict,fea_click_count_dict=get_fea_click_count_dict(train,feat)
    predict[feat+'_stas']=predict[feat].apply(lambda x:func(x,fea_count_dict,fea_click_count_dict))
    return train[[feat+'_stas']],predict[[feat+'_stas']]

def hebing(x,y,z):
    cross=""
    x=str(x)
    y=str(y).split()
    z=str(z).split()
    for i in y:
        for j in z:
            cross+=x+"_"+i+"_"+j+" "
    return cross.strip()

#a b c

#ad_feat_list=['aid','advertiserId','campaignId','creativeSize','adCategoryId','productType','productId']
ad_feat_list=['aid']
#16
user_feat_list=['kw2','topic2','interest1','age','LBS','education']
all_train=None
all_test=None
for afeat in ad_feat_list:
    user_feat_list_len=len(user_feat_list)
    for ufeat_index1 in range(user_feat_list_len-1):#[0-4]
        for ufeat_index2 in range(ufeat_index1+1,user_feat_list_len):#[]
            ufeat1=user_feat_list[ufeat_index1]
            ufeat2=user_feat_list[ufeat_index2]
            feat=afeat+'_'+ufeat1+'_'+ufeat2+'_multi'
            train[feat]=train[[afeat,ufeat1,ufeat2]].apply(lambda x:hebing(x[afeat],x[ufeat1],x[ufeat2]),axis=1)
            predict[feat]=predict[[afeat,ufeat1,ufeat2]].apply(lambda x:hebing(x[afeat],x[ufeat1],x[ufeat2]),axis=1)
            train_,test_=Feature(train,predict,feat)
            all_train=pd.concat([all_train,train_],axis=1)
            all_test=pd.concat([all_test,test_],axis=1)
            print(feat,'done!')
            del train[feat],predict[feat]
            gc.collect()
    del train[afeat],predict[afeat]

all_train.to_csv("aid_user_three_cross_statis_feature_train.csv",index=None)
all_test.to_csv("aid_user_three_cross_statis_feature_test.csv",index=None)

t2=datetime.now()

print(t2-t1)

        
        





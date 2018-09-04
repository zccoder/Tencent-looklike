import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import gc
import os
from datetime import datetime

t1=datetime.now()
ad_feature=pd.read_csv("../data/adFeature.csv")

user_feature=pd.read_csv("../data/userFeature.csv")

train=pd.read_csv("../data/train.csv")
predict=pd.read_csv("../data/test1.csv")

train=pd.merge(train,user_feature,on='uid',how='left')
train=pd.merge(train,ad_feature,on='aid',how='left')
predict=pd.merge(predict,user_feature,on='uid',how='left')
predict=pd.merge(predict,ad_feature,on='aid',how='left')
train.loc[train['label']==-1,'label']=0
train=train.fillna('-1')
predict=predict.fillna('-1')

print(train.shape)
print(predict.shape)

def statis_feat(df,df_val,feature):
    df=df.groupby(feature)["label"].agg(['sum','count']).reset_index()
    new_feat_name=feature+'_stas'
    df.loc[:,new_feat_name]=100*(df['sum']+0.0001)/(df['count']+0.0001)
    df.loc[:,new_feat_name] = np.round(df.loc[:,new_feat_name].values,4)
    df_stas = df[[feature,new_feat_name]]
    df_val=pd.merge(df_val,df_stas,how='left',on=feature)
    return df_val[['index',new_feat_name]]#返回index,new_feat_name

def Feature(train,predict,feat):
    train['index']=list(range(train.shape[0]))
    predict['index']=list(range(predict.shape[0]))
    df_stas_feat=None
    kf=KFold(n_splits=5,random_state=2018,shuffle=True)
    for train_index,val_index in kf.split(train):
        X_train=train.loc[train_index,:]
        X_val=train.loc[val_index,:]

        X_val=statis_feat(X_train,X_val,feat)
        df_stas_feat=pd.concat([df_stas_feat,X_val],axis=0)
    train=pd.merge(train,df_stas_feat,how='left',on='index')
    X_pred=statis_feat(train,predict,feat)
    return train[[feat+'_stas']],X_pred[[feat+'_stas']]

#14
feat_List=['aid','age','gender','education','consumptionAbility','LBS','carrier','house','advertiserId','campaignId','adCategoryId','creativeSize','productType','productId']
all_train=None
all_test=None
for feat in feat_List:
    train_,test_=Feature(train,predict,feat)
    all_train=pd.concat([all_train,train_],axis=1)
    all_test=pd.concat([all_test,test_],axis=1)
    print(feat,'done!')

all_train.to_csv("one_hot_statis_feature_train.csv",index=None)
all_test.to_csv("one_hot_statis_feature_test.csv",index=None)

t2=datetime.now()

print(t2-t1)
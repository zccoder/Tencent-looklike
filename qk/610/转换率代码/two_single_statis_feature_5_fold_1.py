import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import gc
import os
from tqdm import tqdm
from datetime import datetime

t1=datetime.now()
all_vector=['creativeSize','aid','advertiserId','campaignId','creativeId','adCategoryId','productId','productType','LBS','age','appIdAction','appIdInstall',
'carrier','consumptionAbility','ct','education','gender','house','interest1','interest2','interest3',
'interest4','interest5','kw1','kw2','kw3','os','marriageStatus','topic1','topic2','topic3']
#15
usecols=['creativeSize_LBS_stas', 'productType_LBS_stas', 'adCategoryId_LBS_stas', 'advertiserId_LBS_stas', 'aid_LBS_stas', 'productId_LBS_stas', 'advertiserId_consumptionAbility_stas', 'campaignId_LBS_stas', 'aid_consumptionAbility_stas', 'aid_gender_stas', 'adCategoryId_age_stas', 'creativeSize_age_stas', 'campaignId_gender_stas', 'aid_carrier_stas', 'creativeSize_gender_stas', 'adCategoryId_consumptionAbility_stas', 'aid_house_stas', 'aid_age_stas', 'advertiserId_gender_stas','label']
usecols=[i.replace('stas','stats') for i in usecols]

usetruecols=usecols[0:10]
usetruecols.append(usecols[-1])

feat_List=usetruecols[:-1]

train=pd.read_csv("cross_id_content/train_two_single_1.csv",usecols=usetruecols)
test=pd.read_csv("cross_id_content/test_two_single_1.csv",usecols=usetruecols)

print(train.shape)
print(test.shape)

def statis_feat(df,df_val,feature):
    df=df.groupby(feature)["label"].agg(['sum','count']).reset_index()
    new_feat_name=feature+'_stas'
    df.loc[:,new_feat_name]=100*(df['sum']+1+0.0001)/(df['count']+30+0.0001)
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

all_train=None
all_test=None
for feat in tqdm(feat_List):
    train_feat=train[[feat,'label']]
    test_feat=test[[feat]]
    train_,test_=Feature(train_feat,test_feat,feat)
    all_train=pd.concat([all_train,train_],axis=1)
    all_test=pd.concat([all_test,test_],axis=1)
    del train[feat],test[feat]
    gc.collect()
    print(feat,'done!')

all_train.to_csv("two_single_statis_feature_train_1.csv",index=None)
all_test.to_csv("two_single_statis_feature_test_1.csv",index=None)
t2=datetime.now()
print(t2-t1)


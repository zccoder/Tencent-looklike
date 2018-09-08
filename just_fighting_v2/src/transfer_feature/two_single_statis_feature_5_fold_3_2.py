import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import gc
import os
from tqdm import tqdm
from datetime import datetime

t1=datetime.now()
input_path='../../all_feature/combine_feature_file/'
save_path='../../all_feature/transfer_feature_file/'
all_vector=['ct', 'advertiserId', 'carrier', 'aid', 'os', 'age', 'consumptionAbility', 'house', 'gender', 'adCategoryId', 'creativeSize', 'productType', 'campaignId']
#15
usecols=['campaignId_age_stas', 'advertiserId_house_stas','campaignId_ct_stas','aid_os_stas','adCategoryId_ct_stas','advertiserId_age_stas','adCategoryId_carrier_stas','campaignId_consumptionAbility_stas','productType_age_stas','creativeSize_consumptionAbility_stas','creativeSize_carrier_stas','adCategoryId_gender_stas','advertiserId_carrier_stas','adCategoryId_house_stas','label']
print(len(usecols))
usecols=[i.replace('stas','stats') for i in usecols]

usetruecols=usecols[5:10]
usetruecols.append(usecols[-1])

feat_List=usetruecols[:-1]

train=pd.read_csv(input_path+"train_two_single_3_2.csv",usecols=usetruecols)
test=pd.read_csv(input_path+"test_two_single_3_2.csv",usecols=usetruecols)

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

all_train.to_csv(save_path+"two_single_statis_feature_train_3_2.csv",index=None)
all_test.to_csv(save_path+"two_single_statis_feature_test_3_2.csv",index=None)
t2=datetime.now()
print(t2-t1)


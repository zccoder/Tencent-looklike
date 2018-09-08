import pandas as pd
import numpy as np
from datetime import datetime
import gc
from sklearn.model_selection import KFold
from tqdm import tqdm

t1=datetime.now()
input_path='../../data/'
save_path='../../all_feature/transfer_feature_file/'
usecols=['interest1','interest2','interest5','kw2','label']
feat_List=usecols[:-1]

train_test=pd.read_csv(input_path+"train_test_data.csv",usecols=usecols)
train_test=train_test.fillna('-1')
train=train_test[train_test.label!=-1]
test=train_test[train_test.label==-1]
test.reset_index(drop=True,inplace=True)
del train_test
gc.collect()


def Feature_len(train,predict,feat):
	train.loc[:,feat+'_len']=train[feat].apply(lambda x:len(x.split()) if x!='-1' and x!='0' else 0)
	predict.loc[:,feat+'_len']=predict[feat].apply(lambda x:len(x.split()) if x!='-1' and x!='0' else 0)
	return train[[feat+'_len']],predict[[feat+'_len']]

def statis_feat(df,df_val,feature):
    df=df.groupby(feature)["label"].agg(['sum','count']).reset_index()
    new_feat_name=feature+'_stas'
    df.loc[:,new_feat_name]=100*(df['sum']+1+0.0001)/(df['count']+30+0.0001)
    df.loc[:,new_feat_name] = np.round(df.loc[:,new_feat_name].values,4)
    df_stas = df[[feature,new_feat_name]]
    df_val=pd.merge(df_val,df_stas,how='left',on=feature)
    return df_val[['index',new_feat_name]]#返回index,new_feat_name

def Feature(train,predict,feat):
    train.loc[:,'index']=list(range(train.shape[0]))
    predict.loc[:,'index']=list(range(predict.shape[0]))
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
	train_,test_=Feature_len(train,test,feat)
	all_train=pd.concat([all_train,train_],axis=1)
	all_test=pd.concat([all_test,test_],axis=1)
	print(feat,'done!')

feat='interest2'
train_feat=train[[feat,'label']]
test_feat=test[[feat]]
train_,test_=Feature(train_feat,test_feat,feat)
all_train=pd.concat([all_train,train_],axis=1)
all_test=pd.concat([all_test,test_],axis=1)
print(feat,'done!')

all_train.to_csv(save_path+"vector_len_feature_train.csv",index=None)
all_test.to_csv(save_path+"vector_len_feature_test.csv",index=None)

t2=datetime.now()
print(t2-t1)
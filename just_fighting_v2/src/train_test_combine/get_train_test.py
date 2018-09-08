import pandas as pd
input_path='../../data/'
ad_feature=pd.read_csv(input_path+"adFeature.csv")
user_feature=pd.read_csv(input_path+"userFeature.csv")
train=pd.read_csv(input_path+"train.csv")
test=pd.read_csv(input_path+"test.csv")
train.loc[train['label']==-1,'label']=0
test['label']=-1
data=pd.concat([train,test])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
data.to_csv(input_path+"train_test_data.csv",index=None)
print("train_test_concat finished")
train_y=train[['label']]
train_y.to_csv(input_path+"all_train_y.csv",index=None)

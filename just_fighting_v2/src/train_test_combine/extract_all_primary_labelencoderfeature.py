import pandas as pd

input_path='../../data/'
save_path='../../data/'
adFeature_Labelencoder=pd.read_csv(input_path+"adFeature_Labelencoder.csv")
userFeature_Labelencoder=pd.read_csv(input_path+"userFeature_Labelencoder.csv")

train=pd.read_csv(input_path+"train.csv")
train=train[['uid','aid']]
test=pd.read_csv(input_path+"test.csv")

train=pd.merge(train,userFeature_Labelencoder,on='uid',how='left')
train=pd.merge(train,adFeature_Labelencoder,on='aid',how='left')

test=pd.merge(test,userFeature_Labelencoder,on='uid',how='left')
test=pd.merge(test,adFeature_Labelencoder,on='aid',how='left')

train.to_csv(save_path+"train_primary_labelencoder_x.csv",index=None)
test.to_csv(save_path+"test_primary_labelencoder_x.csv",index=None)

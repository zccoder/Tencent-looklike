import pandas as pd
input_path='../../data/'
adFeature_cs=pd.read_csv(input_path+"adFeature_cs.csv")
adFeature_fs=pd.read_csv(input_path+"adFeature_fs.csv")
adFeature=pd.concat([adFeature_cs,adFeature_fs])
adFeature.drop_duplicates(subset='aid',inplace=True)
adFeature.to_csv(input_path+"adFeature.csv",index=None)
print("adFeature concat finished")

userFeature_cs=pd.read_csv(input_path+"userFeature_cs.csv")
userFeature_fs=pd.read_csv(input_path+"userFeature_fs.csv")
userFeature=pd.concat([userFeature_cs,userFeature_fs])
userFeature.drop_duplicates(subset='uid',inplace=True)
userFeature.to_csv(input_path+"userFeature.csv",index=None)
print("userFeature concat finished")

train_cs=pd.read_csv(input_path+"train_cs.csv")
train_fs=pd.read_csv(input_path+"train_fs.csv")
train=pd.concat([train_cs,train_fs])
train.to_csv(input_path+"train.csv",index=None)
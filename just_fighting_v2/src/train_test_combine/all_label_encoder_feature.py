import pandas as pd
from sklearn.preprocessing import LabelEncoder

input_path='../../data/'
save_path='../../data/'
ad_feature=pd.read_csv(input_path+'adFeature.csv')
user_feature=pd.read_csv(input_path+'userFeature.csv')
ad_feature=ad_feature.fillna('-1')
#user_feature=user_feature.fillna('-1')

for fea in ad_feature.columns:#注意aid要保留原始的
	if fea!='aid':
		print(fea,len(ad_feature[fea].value_counts()))
		try:
			ad_feature[fea]=LabelEncoder().fit_transform(ad_feature[fea].apply(int))
		except:
			ad_feature[fea]=LabelEncoder().fit_transform(ad_feature[fea])

ad_feature.to_csv(save_path+"adFeature_Labelencoder.csv",index=None)

for fea in user_feature.columns:#注意uid要保留原始的
	if fea!='uid':
		print(fea,len(user_feature[fea].fillna('-1').value_counts()))
		try:
			user_feature[fea]=LabelEncoder().fit_transform(user_feature[fea].fillna('-1').apply(int))
		except:
			user_feature[fea]=LabelEncoder().fit_transform(user_feature[fea].fillna('-1'))
user_feature.to_csv(save_path+"userFeature_Labelencoder.csv",index=None)
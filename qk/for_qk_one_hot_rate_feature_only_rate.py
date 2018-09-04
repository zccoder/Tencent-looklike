import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import lightgbm as lgb
from scipy import sparse

userFeature=pd.read_csv("../data/userFeature.csv")
ad_feature=pd.read_csv("../data/adFeature.csv")
train=pd.read_csv("../data/train.csv")
data=pd.merge(train,userFeature,on='uid',how='left')
data=pd.merge(data,ad_feature,on='aid',how='left')

predict=pd.read_csv("../data/test1.csv")
predict=pd.merge(predict,userFeature,on='uid',how='left')
predict=pd.merge(predict,ad_feature,on='aid',how='left')

train_Y=data.label

data=data.fillna(-1)

#one_hot=['LBS','age','education','consumptionAbility','carrier','house','aid','advertiserId','campaignId','creativeId','creativeSize','adCategoryId']

train_len=int(train.shape[0]/2)  #421961个正样本 总的有8798814

train_a=data[0:train_len]
train_b=data[train_len:]
allfeature=['aid','age','gender','education','consumptionAbility','LBS','carrier','house','advertiserId','campaignId','creativeId','adCategoryId','creativeSize']

for fea in allfeature:
	print(fea)
	xx=train_a[[fea]]
	xx.drop_duplicates(inplace=True)
	xx1=train_a[['uid',fea]]
	xx1.uid=1
	xx1=xx1.groupby(fea).agg('sum').reset_index()#该广告一共被推送的次数
	xx1.rename(columns={'uid':'count_%s'%fea},inplace=True)
	xx2=train_a[train_a.label==1][['uid',fea]]
	xx2.uid=1
	xx2=xx2.groupby(fea).agg('sum').reset_index()#该广告一共被点击的次数
	xx2.rename(columns={'uid':'count_click_%s'%fea},inplace=True)
	xx_feature=pd.merge(xx,xx1,on=fea,how='left')
	xx_feature=pd.merge(xx_feature,xx2,on=fea,how='left')
	xx_feature["count_%s"%fea]=xx_feature['count_%s'%fea].replace(np.nan,0)
	xx_feature['%s_click_rate'%fea]=xx_feature['count_click_%s'%fea].astype('float')/xx_feature["count_%s"%fea].astype('float')
	train_b=pd.merge(train_b,xx_feature[[fea,'%s_click_rate'%fea]],on=fea,how='left')
	predict=pd.merge(predict,xx_feature[[fea,'%s_click_rate'%fea]],on=fea,how='left')

	xx=train_b[[fea]]
	xx.drop_duplicates(inplace=True)
	xx1=train_b[['uid',fea]]
	xx1.uid=1
	xx1=xx1.groupby(fea).agg('sum').reset_index()#该广告一共被推送的次数
	xx1.rename(columns={'uid':'count_%s'%fea},inplace=True)
	xx2=train_b[train_b.label==1][['uid',fea]]
	xx2.uid=1
	xx2=xx2.groupby(fea).agg('sum').reset_index()#该广告一共被点击的次数
	xx2.rename(columns={'uid':'count_click_%s'%fea},inplace=True)
	xx_feature=pd.merge(xx,xx1,on=fea,how='left')
	xx_feature=pd.merge(xx_feature,xx2,on=fea,how='left')
	xx_feature["count_%s"%fea]=xx_feature['count_%s'%fea].replace(np.nan,0)
	xx_feature['%s_click_rate'%fea]=xx_feature['count_click_%s'%fea].astype('float')/xx_feature["count_%s"%fea].astype('float')
	train_a=pd.merge(train_a,xx_feature[[fea,'%s_click_rate'%fea]],on=fea,how='left')
	
	predict=pd.merge(predict,xx_feature[[fea,'%s_click_rate'%fea]],on=fea,how='left')
	#predict['count_%s'%fea]=((predict['count_%s_x'%fea]+predict['count_%s_y'%fea])/2)
	#predict['count_click_%s'%fea]=((predict['count_click_%s_x'%fea]+predict['count_click_%s_y'%fea])/2)
	predict['%s_click_rate'%fea]=((predict['%s_click_rate_x'%fea]+predict['%s_click_rate_y'%fea])/2)
	predict=predict.drop(['%s_click_rate_x'%fea,'%s_click_rate_y'%fea],axis=1)
	#predict=predict.drop(['count_%s_x'%fea,'count_%s_y'%fea,'count_click_%s_x'%fea,'count_click_%s_y'%fea,'%s_click_rate_x'%fea,'%s_click_rate_y'%fea],axis=1)


train_ab=pd.concat([train_a,train_b])

train_ab=train_ab.drop(['aid', 'uid', 'LBS', 'age', 'appIdAction', 'appIdInstall',
       'carrier', 'consumptionAbility', 'ct', 'education', 'gender', 'house',
       'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1',
       'kw2', 'kw3', 'marriageStatus', 'os', 'topic1', 'topic2', 'topic3',
       'advertiserId', 'campaignId', 'creativeId', 'creativeSize',
       'adCategoryId', 'productId', 'productType','label'],axis=1)
predict=predict.drop(['aid', 'uid', 'LBS', 'age', 'appIdAction', 'appIdInstall',
       'carrier', 'consumptionAbility', 'ct', 'education', 'gender', 'house',
       'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1',
       'kw2', 'kw3', 'marriageStatus', 'os', 'topic1', 'topic2', 'topic3',
       'advertiserId', 'campaignId', 'creativeId', 'creativeSize',
       'adCategoryId', 'productId', 'productType'],axis=1)
train_ab=train_ab.fillna(0)
predict=predict.fillna(0)
#train_ab=sparse.csr_matrix(train_ab)
#predict=sparse.csr_matrix(predict)
#sparse.save_npz("train_single_value_statistics_feature.npz",train_ab)
#sparse.save_npz("test_single_value_statistics_feature.npz",predict)
train_ab.to_csv("train_single_onehotvalue_statistics_feature_only_rate.csv",index=None)
predict.to_csv("test_single_onehotvalue_statistics_feature_only_rate.csv",index=None)
print(train_ab.shape)
print(predict.shape)
import pandas as pd
import numpy as np
import collections
import itertools

userFeature=pd.read_csv("../data/userFeature.csv")
train=pd.read_csv("../data/train.csv")
data=pd.merge(train,userFeature,on='uid',how='left')
ad_feature=pd.read_csv("../data/adFeature.csv")
data=pd.merge(data,ad_feature,on='aid',how='left')

predict=pd.read_csv("../data/test1.csv")
predict=pd.merge(predict,userFeature,on='uid',how='left')
predict=pd.merge(predict,ad_feature,on='aid',how='left')

train_Y=data.label

data=data.fillna("-1")

train_len=int(train.shape[0]/2) 
train_a=data[0:train_len]
train_b=data[train_len:]

allvectorfeature=['marriageStatus','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
def func(x_key_list,fea_count_dict,fea_click_count_dict): ##返回点击的平均值，浏览的平均值，转换率平均值
	x_key_list=str(x_key_list).split()
	count_times=0.0
	click_count_times=0.0
	click_rate=0.0
	for i in x_key_list:
		count_times+=fea_count_dict[str(i)]
		click_count_times+=fea_click_count_dict[str(i)]
	count_times=float(count_times)/len(x_key_list)
	click_count_times=float(click_count_times)/len(x_key_list)
	if count_times==0:
		click_rate=0
	else:
		click_rate=float(click_count_times/count_times)
	return click_rate
    #return count_times,click_count_times,click_rate

for fea in allvectorfeature:
	print(fea)
	###train_a
	fea_count=train_a[fea].apply(lambda x:x.split(' ')).values
	fea_click_count=train_a[train_a.label==1][fea].apply(lambda x:x.split(' ')).values

	fea_count_out = list(itertools.chain.from_iterable(fea_count))
	fea_click_count_out=list(itertools.chain.from_iterable(fea_click_count))

	fea_count_dict=collections.Counter(fea_count_out)
	fea_click_count_dict=collections.Counter(fea_click_count_out)

	train_b['%s_click_rate'%fea]=train_b[fea].apply(lambda x:func(x,fea_count_dict,fea_click_count_dict))
	predict['%s_click_rate_x'%fea]=predict[fea].apply(lambda x:func(x,fea_count_dict,fea_click_count_dict))
	###train_b
	fea_count=train_b[fea].apply(lambda x:x.split(' ')).values
	fea_click_count=train_b[train_b.label==1][fea].apply(lambda x:x.split(' ')).values

	fea_count_out = list(itertools.chain.from_iterable(fea_count))
	fea_click_count_out=list(itertools.chain.from_iterable(fea_click_count))

	fea_count_dict=collections.Counter(fea_count_out)
	fea_click_count_dict=collections.Counter(fea_click_count_out)

	train_a['%s_click_rate'%fea]=train_a[fea].apply(lambda x:func(x,fea_count_dict,fea_click_count_dict))
	predict['%s_click_rate_y'%fea]=predict[fea].apply(lambda x:func(x,fea_count_dict,fea_click_count_dict))

	predict['%s_click_rate'%fea]=((predict['%s_click_rate_x'%fea]+predict['%s_click_rate_y'%fea])/2)
	predict=predict.drop(['%s_click_rate_x'%fea,'%s_click_rate_y'%fea],axis=1)

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

train_ab.to_csv("train_single_vectorvalue_statistics_feature_only_rate.csv",index=None)
predict.to_csv("test_single_vectorvalue_statistics_feature_only_rate.csv",index=None)
print(train_ab.shape)
print(predict.shape)
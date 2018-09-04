
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', 100)
#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 1000)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)
pd.set_option('display.max_columns',None)

import time
import datetime
import gc
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer # 文本向量化
from scipy import sparse


# In[2]:


def get_predict_properties(x):
    p=[] #property:count
    for n in x:        
        if len(n.split(':'))>1:
            for prop in (n.split(':')[1]).split(','):          
                p.append(prop)
    return p 
#观察预测对的个数
def predict_case(threshold = 0.06):
    index = [i for i,x in enumerate(y_pre[:,1]>threshold) if x == 1]
    print(y_test.iloc[index].sum(),(y_pre[:,1]>threshold).sum(),y_test.sum())
    
def save_feature_importance(feat_imp,date):
    feat_imp=sorted(feat_imp.items(),key=lambda x:x[1],reverse=True)
    num = [i+1 for i in range(len(feat_imp)) ]
    features = [i[0] for i in feat_imp ]
    importances = [i[1] for i in feat_imp ]
    feat = pd.DataFrame()
    feat['NO.'] = num
    feat['feature'] = features
    feat['importance'] = importances
    feat.to_csv('./features/feature_importance_'+str(date)+'_zhouc.csv',sep=",",index=False)

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def predict_map(result,num = 166):
    the_prob = np.partition(result,-num)[-num]  #find the 200th largest
    minP = result.min()
    maxP = result.max()
    result=result.copy()
    for i,x in enumerate(result):
        if result[i]<the_prob:
            result[i]=result[i]-minP+1e-10
        else: 
            result[i]=(result[i]-the_prob)/maxP    
    return result
def get_result(pre,num = 40):
    p1 = np.partition(pre[:,0],-num)[-num]  #find the 166th largest
    p2 = np.partition(pre[:,1],-num)[-num]  #find the 166th largest
    result=np.zeros(pre.shape[0])
    for i,x in enumerate(pre):
        if (x[0]>p1 ) & (x[1]>p2):
            result[i]=x.max()
        else: 
            result[i]=x.min()    
    return result


# In[3]:


star_columns= [ 'user_star_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
           'shop_review_num_level', 'shop_star_level']
score_columns= [  'shop_review_positive_rate',
           'shop_score_service', 'shop_score_delivery', 'shop_score_description']


# In[4]:


def convert_data(data):
    category_columns= [  'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
           'user_age_level', 'user_gender_id','user_star_level',  'context_page_id',  
                'shop_review_num_level', 'shop_star_level']
    continuous_columns= ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
    #填充-1  
    for c in category_columns:
        data[c].replace(-1, data[c].mode()[0], inplace=True)   
    for c in continuous_columns:
        data[c].replace(-1, data[c].mean(), inplace=True) 
 ### 4.6 how to use predict_category_property?
    predict_categories= pd.DataFrame(data.predict_category_property.str.split(';').apply(lambda x: [n.split(':')[0] for n in x]))

    predict_categories.predict_category_property = predict_categories.predict_category_property.apply(lambda x: set(x))
    
    item_categories = pd.DataFrame(data.item_category_list.str.split(';').tolist()).add_prefix('category_')
    onehot_category = pd.get_dummies(item_categories)
    data = pd.concat([data, onehot_category], axis=1) 
    
    item_categories = pd.concat([item_categories, predict_categories], axis=1)
    data['predict_category_0']=item_categories.apply(lambda row:  row.category_0 in row.predict_category_property, axis=1)
    data['predict_category_1']=item_categories.apply(lambda row:  row.category_1 in row.predict_category_property, axis=1)
    data['predict_category_2']=item_categories.apply(lambda row:  row.category_2 in row.predict_category_property, axis=1)


#     data.insert(loc=0, column='predict_category_0', value=item_categories.apply(lambda row:  row.category_0 in row.predict_category_property, axis=1))
#     data.insert(loc=0, column='predict_category_1', value=item_categories.apply(lambda row:  row.category_1 in row.predict_category_property, axis=1))
#     data.insert(loc=0, column='predict_category_2', value=item_categories.apply(lambda row:  row.category_2 in row.predict_category_property, axis=1))

    predict_properties= pd.DataFrame(data.predict_category_property.str.split(';').apply(get_predict_properties))

    item_properties = pd.DataFrame(data.item_property_list.str.split(';'))
    item_properties = pd.concat([item_properties, predict_properties], axis=1)

    right_properties=item_properties.apply(lambda row:  set(row.item_property_list).intersection(set(row.predict_category_property)), axis=1)

    right_properties = pd.DataFrame(right_properties,columns = ['right_properties'])

    data.insert(loc=0, column='right_prop_num', value=right_properties.right_properties.apply(lambda x: len(x)))

    item_properties = pd.concat([item_properties, right_properties], axis=1)   
# 广告预测准确率predict_category_property
    prop_recall = item_properties.apply(lambda row: len(row.right_properties)/len(row.item_property_list), axis=1)

    prop_precision = item_properties.apply(lambda row: len(row.right_properties)/len(row.predict_category_property) if len(row.predict_category_property)>0 else 0, axis=1)

    data.insert(loc=0, column='prop_recall', value=prop_recall)
    data.insert(loc=0, column='prop_precision', value=prop_precision)
      
    dummy_columns= [ 'predict_category_0','predict_category_1']#,'predict_category_2']
    data = pd.get_dummies(data,columns=dummy_columns)  #会删除原有的列    
    #     data.shop_score_delivery=pd.qcut(data.shop_score_delivery, 10, labels=False)  #0.08057535152689949  
    #from 1
    data.user_age_level=data.user_age_level- 999
    data.context_page_id=data.context_page_id - 4000
    #from 0
    data.user_star_level=data.user_star_level- 3000
    data.shop_star_level=data.shop_star_level-4999
    return data


# In[6]:


online = True  # 这里用来标记是 线下验证 还是 在线提交
data = pd.read_csv('./round2_train.txt', sep=' ')
testA = pd.read_csv('./round2_ijcai_18_test_a_20180425.txt', sep=' ')
testB = pd.read_csv('./round2_ijcai_18_test_b_20180510.txt', sep=' ')
testB_instance_id=testB.instance_id
test= pd.concat([testA, testB],ignore_index=True)
del testA,testB
data = pd.concat([data, test],ignore_index=True)
    # 广告被点击时间 day hour
data['time'] = data.context_timestamp.apply(timestamp_datetime)
data['day'] = data.time.apply(lambda x: int(x[8:10]))
data['hour'] = data.time.apply(lambda x: int(x[11:13]))

data = convert_data(data)


# In[7]:


data['item_property_list']=data.item_property_list.apply(lambda x: ' '.join(x.split(';')))
cv=CountVectorizer(max_features=50) #取频率最高的top 50
data_a = cv.fit_transform(data['item_property_list'])
# data = sparse.hstack((data[features], data_a))
data_a = pd.DataFrame(data_a.todense(),columns=['p_'+str(i) for i in range(data_a.shape[1])])
data = pd.concat([data,data_a],axis=1)


# In[8]:


# cv=CountVectorizer(max_features=50) #取频率最高的top 50
# data_a = cv.fit_transform(data['item_city_id'].astype(str))
# data_a = pd.DataFrame(data_a.todense(),columns=['city_'+str(i) for i in range(data_a.shape[1])])
# data = pd.concat([data,data_a],axis=1)
del data_a
gc.collect()


# In[9]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
def stringColumn2int(data):
    feat_set = ['item_category_list','item_property_list','predict_category_property']
    lbe=LabelEncoder()   
    #进行encoder
    for col in feat_set:
        data[col]=lbe.fit_transform(data[col])
    return data
stringColumn2int(data)
gc.collect()


# In[10]:


def add_features(data):    
# 原始特征 过拟合 1 已有 2

# 用户相关的特征
# 用户线上点击次数
    data['user_count'] = data.groupby(['user_id']).instance_id.transform('count')
# 用户点击的每一商家次数，及其占所有总数的比重
    data['user_count_shop'] = data.groupby(['user_id','shop_id']).instance_id.transform('count') 
    data['user_shop_rate'] = data['user_count_shop']/data['user_count'] 
# 用户点击的不同商家数
    data['user_countU_shop'] = data.groupby(['user_id']).shop_id.transform('nunique') 
# 用户点击的不同item次数，及其占总数的比重
    data['user_count_item'] = data.groupby(['user_id','item_id']).instance_id.transform('count') 
    data['user_item_rate'] = data['user_count_item']/data['user_count']
# 用户点击的不同item数
    data['user_countU_item'] = data.groupby(['user_id']).item_id.transform('nunique') 
# 用户平均点击商家次数
    data['user_shop_mean'] = data.groupby(['user_id']).user_count_shop.transform('mean') 
# 用户平均点击item次数
    data['user_item_mean'] = data.groupby(['user_id']).user_count_item.transform('mean') 
# 用户点击的不同类目数、品牌数、城市、页数、天数等
    data['user_countU_category'] = data.groupby(['user_id']).item_category_list.transform('nunique') 
    data['user_countU_brand'] = data.groupby(['user_id']).item_brand_id.transform('nunique') 
    data['user_countU_city'] = data.groupby(['user_id']).item_city_id.transform('nunique')
    data['user_countU_page'] = data.groupby(['user_id']).context_page_id.transform('nunique')
    data['user_countU_day'] = data.groupby(['user_id']).day.transform('nunique') 
    data['user_countU_preprop'] = data.groupby(['user_id']).predict_category_property.transform('nunique') 
# 用户点击排序    
    data['user_click_rank'] = data.groupby(['user_id'])['context_timestamp'].rank(pct=True)      
    data['user_cate_rank'] = data.groupby(['user_id','item_category_list'])['context_timestamp'].rank(pct=True)     
    data['user_item_rank'] = data.groupby(['user_id','item_id'])['context_timestamp'].rank(pct=True)
    data['user_shop_rank'] = data.groupby(['user_id','shop_id'])['context_timestamp'].rank(pct=True)
    data['user_preprop_rank'] = data.groupby(['user_id','predict_category_property'])['context_timestamp'].rank(pct=True)     
# 用户是否已买过 尝试过不明显   
    data['one_record_oneday']=(data.user_count==1) & (data.user_countU_day==1)
    data['some_record_oneday']=(data.user_count!=1) & (data.user_countU_day==1)
    data['some_record_manyday']=(data.user_count!=1) & (data.user_countU_day!=1)
# 用户当天点击次数，及其占总数比重
    data['user_count_day'] = data.groupby(['user_id','day']).instance_id.transform('count') 
    data['user_day_rate'] = data['user_count_day']/data['user_count'] 

# 商家相关的特征
# 商家被点击次数
    data['shop_click_count'] = data.groupby(['shop_id']).instance_id.transform('count')
# 某类目商家被点击次数，及其占总数比重
    data['cate_count_shop'] = data.groupby(['item_category_list','shop_id']).instance_id.transform('count') 
    data['cate_shop_rate'] = data['cate_count_shop']/data['shop_click_count'] 
# 商家被购买的平均时间率
# 商家被核销过的不同item数量
    data['shop_count_item'] = data.groupby(['shop_id','item_id']).instance_id.transform('count') 
# 商家被核销过的不同item数量占所有被点击不同item数量的比重
    data['shop_item_rate'] = data['shop_count_item']/data['shop_click_count']
# 商家不同类目数、item数、城市、页数、天数等
    data['shop_countU_category'] = data.groupby(['shop_id']).item_category_list.transform('nunique') 
    data['shop_countU_brand'] = data.groupby(['shop_id']).item_brand_id.transform('nunique') 
    data['shop_countU_item'] = data.groupby(['shop_id']).item_id.transform('nunique')
# 商家当天点击次数，及其占总数比重
    data['day_count_shop'] = data.groupby(['day','shop_id']).instance_id.transform('count') 
    data['shop_day_rate'] = data['day_count_shop']/data['shop_click_count'] 
# 商家被多少不同用户点击的数目
    data['shop_countU_user'] = data.groupby(['shop_id']).user_id.transform('nunique')
    data['userU_shop_rate'] = data['shop_countU_user']/data['shop_click_count'] 
    
# 商品相关的特征
# 商品被点击次数
    data['item_click_count'] = data.groupby(['item_id']).instance_id.transform('count')
# 商品当天点击次数，及其占总数比重
    data['day_count_item'] = data.groupby(['day','item_id']).instance_id.transform('count') 
    data['item_day_rate'] = data['day_count_item']/data['item_click_count'] 
# 某类目商品被点击次数，及其占总数比重
    data['cate_count_item'] = data.groupby(['item_category_list','item_id']).instance_id.transform('count') 
    data['cate_item_rate'] = data['cate_count_item']/data['item_click_count'] 
# 商品不同shop数、品牌、页数等
#     data['item_countU_shop'] = data.groupby(['item_id']).shop_id.transform('nunique')
    data['item_countU_page'] = data.groupby(['item_id']).context_page_id.transform('nunique')
    data['item_countU_brand'] = data.groupby(['item_id']).item_brand_id.transform('nunique')
# 价格销量级别比率
    data['price_sale_rate']= data.item_price_level/data.item_sales_level
    data['sale_star_rate']= data.item_sales_level/(data.shop_star_level+1)
    data['sale_pv_rate']= data.item_sales_level/(data.item_pv_level+1)
# 商品当天点击次数，及其占总数比重
    data['day_count_item'] = data.groupby(['day','item_id']).instance_id.transform('count') 
    data['item_day_rate'] = data['day_count_item']/data['item_click_count'] 
# 商品被多少不同用户点击的数目 
    data['item_countU_user'] = data.groupby(['item_id']).user_id.transform('nunique')
    data['userU_item_rate'] = data['item_countU_user']/data['item_click_count'] 
    
# 广告被浏览时间
    data['user_period'] = data.groupby(['user_id']).context_timestamp.transform(lambda x: x.max()-x.min())/60
    data['user_period_oneday'] = data.groupby(['user_id','day']).context_timestamp.transform(lambda x: x.max()-x.min())/60
    data['user_period_mean'] =data['user_period'] /data['user_count']/data['user_countU_category']
    data['user_item_period'] = data.groupby(['user_id','item_id']).context_timestamp.transform(lambda x: x.max()-x.min())/60
    data['user_cate_period'] = data.groupby(['user_id','item_category_list']).context_timestamp.transform(lambda x: x.max()-x.min())/60
# 用户上/下一次点击的时间间隔 
    data['user_click_period_last'] = data.sort_values('context_timestamp').groupby(['user_id']).context_timestamp.agg('diff')/60 
    data['user_click_period_next'] = data.sort_values('context_timestamp').groupby(['user_id']).user_click_period_last.transform(lambda x: x.shift(periods =-1))
    data['click_period_std'] = data.groupby(['user_id','day']).user_click_period_next.transform('std') 
    
# 这部分特征利用了赛题leakage，都是在预测区间提取的。
# 用户总点击次数  2
# 用户点击的不同商家数目 2
# 用户当天点击次数 2
# 用户点击的所有category数目 2
# 用户点击的所有item数目 2
# 商家被多少不同用户点击的数目 2
# 商家的所有item数目 2
    data['item_sale_rank'] = data.groupby(['item_category_list','day'])['item_sales_level'].rank(pct=True) 
    data['shop_star_rank'] = data.groupby(['item_category_list'])['shop_star_level'].rank(pct=True)
    data['item_price_rank'] = data.groupby(['item_category_list'])['item_price_level'].rank(pct=True)
# 用户消费特征 需要先groupby算train得到series，然后map到test（组合购买率没考虑过）

    data['star_std'] = data[star_columns].std(axis=1)
    data['score_std'] = data[score_columns].std(axis=1)
    data['star_sum'] = data[star_columns].sum(axis=1)-data['context_page_id']
    data['score_sum'] = data[score_columns].sum(axis=1)-data['context_page_id']
    data['star_prod'] = data[star_columns].prod(axis=1)
    data['score_prod'] = data[score_columns].prod(axis=1)
    data['item_score'] = data[['item_sales_level', 'item_collected_level', 'item_pv_level']].prod(axis=1)/data['context_page_id']
    data['shop_score'] =data[['shop_score_service', 'shop_score_delivery', 'shop_score_description']].prod(axis=1)
    data['total_score'] = data['score_prod']/data['context_page_id']
    data['total_star'] = data['star_prod']/data['context_page_id']
   #0412 
    data['good_review_num']= data.item_sales_level*data.shop_review_positive_rate
    data['bad_review_num']= data.item_sales_level*(1-data.shop_review_positive_rate)
    data['shop_good_review_num']= data.shop_review_num_level*data.shop_review_positive_rate
    data['shop_bad_review_num']= data.shop_review_num_level*(1-data.shop_review_positive_rate)

    data['brand_click_count'] = data.groupby(['item_brand_id']).instance_id.transform('count')
    data['city_click_count'] = data.groupby(['item_city_id']).instance_id.transform('count')
    data['brand_countU_item'] = data.groupby(['item_brand_id']).item_id.transform('nunique')
    data['city_countU_item'] = data.groupby(['item_city_id']).item_id.transform('nunique')
    data['brand_countU_shop'] = data.groupby(['item_brand_id']).shop_id.transform('nunique')
    data['city_countU_shop'] = data.groupby(['item_city_id']).shop_id.transform('nunique')

    #data.drop(['page_rate'],inplace=True,axis=1)

#     data['user_countU_prop'] = data.groupby(['user_id']).item_property_list.transform('nunique') 
#     data['user_cate_countU_item']=data.groupby(['user_id','item_category_list']).item_id.transform('nunique')
    data['item_price_mean']=data.groupby(['user_id']).item_price_level.transform('mean') 
    data['item_sales_mean']=data.groupby(['user_id']).item_sales_level.transform('mean')
    data['shop_delivery_mean']=data.groupby(['user_id']).shop_score_delivery.transform('mean') 
    data['shop_description_mean']=data.groupby(['user_id']).shop_score_description.transform('mean')
#     data['shop_star_mean']=data.groupby(['user_id']).shop_star_level.transform('mean') 
#     data['shop_service_mean']=data.groupby(['user_id']).shop_score_service.transform('mean') 
# data['item_score'] =data['item_score'].transform(lambda x:(x - min(x))/(max(x)-min(x)))
# data['shop_score']=data['shop_score'].transform(lambda x:(x - min(x))/(max(x)-min(x)))
    # 用户此次之前/后点击次数
    data['user_times_last'] = data.groupby(['user_id','day'])['context_timestamp'].rank()-1
    data['user_times_next'] = data.groupby(['user_id','day'])['context_timestamp'].rank(ascending=False)-1

    data['user_item_times_last'] = data.groupby(['user_id','item_id','day'])['context_timestamp'].rank()-1
    data['user_item_times_next'] = data.groupby(['user_id','item_id','day'])['context_timestamp'].rank(ascending=False)-1

    data['user_cate_times_last'] = data.groupby(['user_id','item_category_list','day'])['context_timestamp'].rank()-1
    data['user_cate_times_next'] = data.groupby(['user_id','item_category_list','day'])['context_timestamp'].rank(ascending=False)-1
    
    return data


# In[11]:


data = add_features(data)
print('add_features time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


# In[12]:


#交叉特征
user_columns= [ 'user_gender_id','user_age_level','user_occupation_id','user_star_level']
item_columns= ['item_category_list', 'item_property_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level' ]
shop_columns= [  'shop_review_num_level', 'shop_star_level']
for f1 in item_columns:
    for f2 in user_columns:
        data[f1+'_'+f2+'_portion'] = data.groupby([f1,f2]).instance_id.transform('count') /data.groupby([f1]).instance_id.transform('count')
for f1 in item_columns:
    for f2 in shop_columns:
        data[f1+'_'+f2+'_portion'] = data.groupby([f1,f2]).instance_id.transform('count') /data.groupby([f1]).instance_id.transform('count')


# In[7]:


#交叉特征(当天)
user_columns= [ 'user_gender_id','user_age_level','user_occupation_id','user_star_level']
item_columns= ['item_category_list', 'item_property_list', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level' ]
shop_columns= [  'shop_review_num_level', 'shop_star_level']
for f1 in item_columns:
    for f2 in user_columns:
        data[f1+'_'+f2+'_portion_day'] = data.groupby([f1,f2,'day']).instance_id.transform('count') /data.groupby([f1,'day']).instance_id.transform('count')
for f1 in item_columns:
    for f2 in shop_columns:
        data[f1+'_'+f2+'_portion_day'] = data.groupby([f1,f2,'day']).instance_id.transform('count') /data.groupby([f1,'day']).instance_id.transform('count')


# In[13]:


data['page_sale_rank'] = data.groupby(['context_page_id','predict_category_property'])['item_sales_level'].rank(pct=True)
data['page_price_rank'] = data.groupby(['context_page_id','predict_category_property'])['item_price_level'].rank(pct=True)

data['page_shop_rank'] = data.groupby(['context_page_id','predict_category_property'])['shop_star_level'].rank(pct=True)
data['page_review_rank'] = data.groupby(['context_page_id','predict_category_property'])['shop_review_num_level'].rank(pct=True)


# In[14]:


# 0 pre_category 用户产生这条记录前，是否产生过相同category的浏览记录
# 1 pre_shopid 用户产生这条记录前，是否产生过相同shop的浏览记录
# 2 pre_itemid 用户产生这条记录前，是否产生过相同item的浏览记录
data['pre_category_clicked']= data.groupby(['user_id','item_category_list'])['context_timestamp'].rank().apply(lambda x: 1 if x>1 else 0)
data['pre_shopid_clicked']= data.groupby(['user_id','shop_id'])['context_timestamp'].rank().apply(lambda x: 1 if x>1 else 0)
data['pre_itemid_clicked']= data.groupby(['user_id','item_id'])['context_timestamp'].rank().apply(lambda x: 1 if x>1 else 0)

# 用户点击的每一商家次数，及其占所有总数的比重
data['user_count_category'] = data.groupby(['user_id','item_category_list']).instance_id.transform('count') 
data['user_cate_rate'] = data['user_count_category']/data['user_count'] 

data['category_click_rank'] = data.groupby(['user_id'])['user_count_category'].rank(pct=True) 
data['item_click_rank'] = data.groupby(['user_id'])['user_count_item'].rank(pct=True) 
data['shop_click_rank'] =data.groupby(['user_id'])['user_count_shop'].rank(pct=True) 


# In[8]:


data.day.replace(31,0,inplace=True) #31号替换成0号，day变成0-7，便于处理
def online_sliding_be_one_click_buy_convert(all_data, first_fe):
    group_list = [first_fe]
    
    iter_list = [7,6,5,4,3,2,1,0]
    # iter_list = [0,6,5,4,3]
    iter_range = list(set(iter_list))
    iter_range.sort(key = iter_list.index)
        
    for i in iter_range:
        sliding_temp = all_data[(all_data.day.isin([i]))&(all_data.hour < 12)][group_list+['is_trade']].reset_index(drop = True)
        sliding_temp['sliding_2d_be_'+first_fe+'_click'] = 1
        sliding_temp.rename(columns={'is_trade': 'sliding_2d_be_' + first_fe + '_buy'}, inplace=True)
        sliding_temp = sliding_temp.groupby(group_list).agg('sum').reset_index()
        
        sliding_temp_hour = all_data[all_data.day.isin([i])][group_list+['hour','is_trade']].reset_index(drop = True)
        sliding_temp_hour['sliding_2d_be_hour'+first_fe+'_click'] = 1
        sliding_temp_hour.rename(columns={'is_trade': 'sliding_2d_be_hour' + first_fe + '_buy'}, inplace=True)

        sliding_temp_hour = sliding_temp_hour.groupby(group_list+['hour']).agg('sum').reset_index()
        sliding_temp_hour = pd.merge(sliding_temp_hour,sliding_temp,how = 'left', on = first_fe)
        sliding_temp_hour['be_' + first_fe + '_convert'] = sliding_temp_hour.apply(lambda x:helper_sliding_beconvert(x['sliding_2d_be_hour' + first_fe + '_buy'],x['sliding_2d_be_hour'+first_fe+'_click'],x['sliding_2d_be_' + first_fe + '_buy'],x['sliding_2d_be_'+first_fe+'_click'],x.hour),axis = 1)
        
        sliding_temp_hour['day'] = i
        
        sliding_temp_hour = sliding_temp_hour[['day',first_fe,'hour','be_' + first_fe + '_convert']].reset_index(drop = True)
        if i == 7:
            new_all_data = sliding_temp_hour.reset_index(drop = True)
        else:
            new_all_data = new_all_data.append(sliding_temp_hour,ignore_index = True)
    print("over.....")
    return new_all_data

def helper_sliding_beconvert(x,y,all_x,all_y,k):
    if k < 12:
        if all_y - y == 0:
            return -1
        else:
            return (all_x - x) / (all_y - y)
    else:
        if all_y == 0 or np.isnan(all_y):
            return -1
        else:
            return all_x / all_y

def add_sliding_be_one_click_buy_convert(pd_data,all_data,first_fe):
    temp_all_data = all_data.drop_duplicates().reset_index(drop = True)
    pd_data = pd.merge(pd_data, temp_all_data, on = ['day','hour',first_fe], how = 'left')
    pd_data['be_' + first_fe + '_convert'].fillna(-1,inplace = True)
    return pd_data


# In[9]:


time_start=time.time()
be_feature_name_convert =['item_id','item_city_id','user_occupation_id','user_star_level', 'user_gender_id','user_age_level','shop_id','item_brand_id']
for i in be_feature_name_convert:
    sliding_convert_data = online_sliding_be_one_click_buy_convert(data,i)
    data = add_sliding_be_one_click_buy_convert(data, sliding_convert_data, i)
time_end=time.time()
print('totally cost: %d s' % (time_end-time_start))


# ## 内存优化

# In[15]:


# data.drop(['time'],axis=1,inplace=True)


# In[10]:


all_data_convert_int = data.select_dtypes(include = ['int64']).apply(pd.to_numeric, downcast = 'unsigned')
data[all_data_convert_int.columns] = all_data_convert_int
del all_data_convert_int
all_data_convert_float = data.select_dtypes(include = ['float64']).apply(pd.to_numeric, downcast = 'float')
data[all_data_convert_float.columns] = all_data_convert_float
del all_data_convert_float
gc.collect()


# In[11]:


data.info(memory_usage='deep')


# In[ ]:


data.to_csv('./alldata_round2_b_min_ffd.txt',sep=' ',index=False, line_terminator='\r', float_format='%g') #内存优化后 无转化率
# data = pd.read_csv('./alldata_round2_b_min.txt',sep=' ')


# In[12]:


print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


# ## 线上

# In[13]:


popFeat = ['instance_id','item_id','item_brand_id', 'item_city_id','shop_id', 'item_category_list', 'item_property_list',
          'context_timestamp', 'context_id','predict_category_property',
           'time','user_id','is_trade','shop_score','item_countU_shop',
#            'user_countU_prop','user_cate_countU_item','context_page_id',
#            'item_price_mean','item_sales_mean','shop_delivery_mean','shop_description_mean','shop_star_mean','shop_service_mean',
#          'star_std', 'score_std', 'star_sum', 'score_sum', 'star_prod', 'score_prod',
           'user_times_last','user_times_next',
           'user_item_times_last','user_item_times_next',
          # 'user_cate_times_last','user_cate_times_next',
#                       'good_review_num','bad_review_num','shop_good_review_num','shop_bad_review_num',
           'brand_click_count','city_click_count','brand_countU_item','city_countU_item','brand_countU_shop','city_countU_shop',
           'day_count_shop','user_day_rate','user_shop_rank','shop_day_rate'
          ]


# In[14]:


features = [f for f in data.columns if f not in popFeat]
gc.collect()
len(features)#290


# In[15]:


era_params={'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'logloss',
	    'gamma':0.1, 
	    'min_child_weight':1.2, #1.1
	    'max_depth':5,  #4
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.03,
	    'tree_method':'hist',
	    'seed':0,
        'silent':1,
	    'nthread':16
	    }
best_iteration = 2000


# In[16]:


test = data.loc[data.is_trade.isnull()]
train = data.loc[(data.day!=6)&(data.is_trade.notnull())]
label=train.is_trade
Dtrain_data = xgb.DMatrix(train[features],label=label)
Dtest_data = xgb.DMatrix(test[features])
print('xgb data loaded!')
sub2 = pd.DataFrame()
sub2['instance_id'] = list(test.instance_id)
del train,test
gc.collect()
time.sleep(60)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


# In[17]:


time_start=time.time()
model = xgb.train(era_params,Dtrain_data,num_boost_round=best_iteration)
print(log_loss(label,model.predict(Dtrain_data)))
test_result = model.predict(Dtest_data)
sub2['predicted_score'] = list(test_result)
#sub2.to_csv('./result/20180511.txt',sep=' ',index=False, line_terminator='\r')
#print(sub2.predicted_score.describe(),sub2.predicted_score.quantile(0.964))
time_end=time.time()
sub2.index = sub2.instance_id
sub = pd.DataFrame()
sub['instance_id'] = list(testB_instance_id)
sub['predicted_score'] = list(sub2['predicted_score'].loc[list(testB_instance_id)].values)
sub.to_csv('./result/20180512_o.txt',sep=' ',index=False, line_terminator='\r')
print(sub.predicted_score.describe(),sub.predicted_score.quantile(0.964))
print('totally cost: %d s' % (time_end-time_start))


# In[ ]:


save_feature_importance(model.get_fscore(),'0512_o')


# In[ ]:





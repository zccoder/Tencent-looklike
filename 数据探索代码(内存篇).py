import pandas as pd
import numpy as np
import gc
from datetime import datetime
t1=datetime.now()
train_test1_2_data=pd.read_csv("train_test1_2_data.csv")
t2=datetime.now()
print(t2-t1)#10:13
print(train_test1_2_data.columns)
#一共33个特征
'''['aid', 'uid', 'label', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'LBS',
       'age', 'appIdAction', 'appIdInstall', 'carrier', 'consumptionAbility',
       'ct', 'education', 'gender', 'house', 'interest1', 'interest2',
       'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3',
       'marriageStatus', 'os', 'topic1', 'topic2', 'topic3']'''
print(train_test1_2_data.shape)#训练集+测试集 68996077,33
print(train_test1_2_data.info(memory_usage='deep'))#dtypes: float64(3), int64(14), object(16)
#memory usage: 71.3 GB  csv文件大小读取19.41G
# aid                   int64
# uid                   int64
# label                 int64
# advertiserId          int64
# campaignId            int64
# creativeId            int64
# creativeSize          int64
# adCategoryId          int64
# productId             int64
# productType           int64
# LBS                   float64
# age                   int64
# appIdAction           object
# appIdInstall          object
# carrier               int64
# consumptionAbility    int64
# ct                    object
# education             int64
# gender                float64
# house                 float64
# interest1             object
# interest2             object
# interest3             object
# interest4             object
# interest5             object
# kw1                   object
# kw2                   object
# kw3                   object
# marriageStatus        object
# os                    object
# topic1                object
# topic2                object
# topic3                object

for dtype in ['float','int','object']:
	selected_dtype=train_test1_2_data.select_dtypes(include=[dtype])
	usage_gb=selected_dtype.memory_usage(deep=True)/1024**3
	mean_usage_gb=usage_gb.mean()
	sum_usage_gb=usage_gb.sum()
	print(dtype+":",mean_usage_gb,sum_usage_gb)
# float: 0.385545643046 1.54218257219  3个float bug 1.54/4了 每个float都是0.514g 因为都是float64
#因为float64是8个字节，8*68996077/1024**3=0.51406GB
# int: 0.479790115356 7.19685173035    14个int  bug 7.19/15了 每个int都占用0.514g 因为都是int64
# object: 3.67974655012 62.5556913521  16个object
# appIdAction       2.131145e+00
# appIdInstall      2.999218e+00
# ct                3.991065e+00
# interest1         6.014488e+00
# interest2         3.744143e+00
# interest3         2.124775e+00
# interest4         2.090417e+00
# interest5         6.321893e+00
# kw1               5.331706e+00
# kw2               5.385227e+00
# kw3               2.210314e+00
# marriageStatus    3.856323e+00
# os                4.211848e+00  五个唯一值
# topic1            4.899926e+00
# topic2            5.046098e+00
# topic3            2.197103e+00
for dtype in ['float','int','object']:
	selected_dtype=train_test1_2_data.select_dtypes(include=[dtype])
	usage_gb=selected_dtype.memory_usage(deep=True)/1024**3
	print(usage_gb)
	mean_usage_gb=usage_gb.mean()
	sum_usage_gb=usage_gb.sum()
	print(mean_usage_gb,sum_usage_gb)

select_int=train_test1_2_data.select_dtypes(include=['int'])
#downcast 
#integer or signed min:np.int8
#unsigned: min:np.uint8
#float: min:np.float32
converted_int=select_int.apply(pd.to_numeric,downcast='integer')
print(converted_int.info(memory_usage='deep'))#原来int 7.19G减少到2G #减少内存72%+
#14
# aid                   int16
# uid                   int32
# label                 int8
# advertiserId          int32
# campaignId            int32
# creativeId            int32
# creativeSize          int8
# adCategoryId          int16
# productId             int32
# productType           int8
# age                   int8
# carrier               int8
# consumptionAbility    int8
# education             int8
# dtypes: int16(2), int32(5), int8(7)
# memory usage: 2.0 GB
select_float=train_test1_2_data.select_dtypes(include=['float'])
converted_float=select_float.apply(pd.to_numeric,downcast='float')
print(converted_float.info(memory_usage='deep'))#从原来的1.54G降低到0.7896G #降低50%
#3
# LBS       float32
# gender    float32
# house     float32
# dtypes: float32(3)
# memory usage: 789.6 MB
train_test1_2_data[converted_int.columns]=converted_int
train_test1_2_data[converted_float.columns]=converted_float
del converted_int
del converted_float
gc.collect()
train_test1_2_data.info(memory_usage='deep')#71.3G下降到65.3G

#os列就五个唯一值，转变为category占用极少的内存
converted_obj=pd.DataFrame()
converted_obj.loc[:,'os']=train_test1_2_data['os'].astype('category')
converted_obj['os'].memory_usage(deep=True)/1024**3 #4.211G
train_test1_2_data['os'].memory_usage(deep=True)/1024**3 #0.06425G

for col in train_test1_2_data.select_dtypes(include=['object']).columns:
	num_unique_values=len(train_test1_2_data[col].unique())
	num_total_values=len(train_test1_2_data[col])
	print(col,num_unique_values,num_unique_values/num_total_values)
# appIdAction 557245 0.008076473681250022
# appIdInstall 867234 0.012569323325440663
# ct 65 9.420825476787615e-07
# interest1 24075965 0.3489468683849953
# interest2 4209642 0.06101277323346949
# interest3 318 4.6089576947976335e-06
# interest4 316 4.57997053948444e-06
# interest5 28068795 0.4068172600595828
# kw1 39099696 0.5666944803253089
# kw2 30401410 0.44062519670502426
# kw3 1255573 0.018197744779025624
# marriageStatus 28 4.058201743846973e-07
# os 5 7.246788828298165e-08
# topic1 39164627 0.5676355628161294
# topic2 22805265 0.3305298792567583
# topic3 1696634 0.02459029663382166

#查看缺失值比例
train_test_len=train_test1_2_data.shape[0]
for col in train_test1_2_data.columns:
	nulllen=train_test1_2_data[col].isnull().sum()
	print(col,nulllen,nulllen/train_test_len)
# aid 0 0.0
# uid 0 0.0
# label 0 0.0
# advertiserId 0 0.0
# campaignId 0 0.0
# creativeId 0 0.0
# creativeSize 0 0.0
# adCategoryId 0 0.0
# productId 0 0.0
# productType 0 0.0
# LBS 326 4.72490631605e-06
# age 0 0.0
# appIdAction 67926065 0.984491697985 多
# appIdInstall 67588053 0.979592694814 多
# carrier 0 0.0
# consumptionAbility 0 0.0
# ct 0 0.0
# education 0 0.0
# gender 2 2.89871553132e-08
# house 56843264 0.823862261038 多
# interest1 6228333 0.0902708280066
# interest2 23997606 0.347811166133
# interest3 67154069 0.973302714008 多
# interest4 67912999 0.984302324899 多
# interest5 17201276 0.249308029499
# kw1 6864095 0.0994852939248
# kw2 2524462 0.0365884860381
# kw3 65848416 0.95437913086 多
# marriageStatus 8 1.15948621253e-07
# os 0 0.0
# topic1 5858756 0.084914335057
# topic2 2639447 0.038255030065
# topic3 65840316 0.954261732881 多

#不同值占的比例不大，所以转换速度还可以
converted_obj=pd.DataFrame()
converted_obj.loc[:,'appIdAction']=train_test1_2_data['appIdAction'].astype('category')
converted_obj['appIdAction'].memory_usage(deep=True)/1024**3 #0.338G
train_test1_2_data['appIdAction'].memory_usage(deep=True)/1024**3 #2.131G

#不同值占的比例太大了，所以转换速度很慢，选择不转换,而且影响不大
converted_obj=pd.DataFrame()
converted_obj.loc[:,'kw1']=train_test1_2_data['kw1'].astype('category')
converted_obj['kw1'].memory_usage(deep=True)/1024**3 #4.7638
train_test1_2_data['kw1'].memory_usage(deep=True)/1024**3 #5.33

# converted_obj=pd.DataFrame()

for col in train_test1_2_data.select_dtypes(include=['object']).columns:
	t1=datetime.now()
	num_unique_values=len(train_test1_2_data[col].unique())
	num_total_values=len(train_test1_2_data[col])
	if num_unique_values/num_total_values<0.1:
		train_test1_2_data.loc[:,col]=train_test1_2_data[col].astype('category')
		print('category:',col,train_test1_2_data[col].memory_usage(deep=True)/1024**3)
	else:
		#train_test1_2_data.loc[:,col]=train_test1_2_data[col]
		print(col,train_test1_2_data[col].memory_usage(deep=True)/1024**3)
	t2=datetime.now()
	print(t2-t1)

# category: appIdAction 0.3381480174139142
# 0:00:45.806129
# category: appIdInstall 0.9288910599425435
# 0:01:56.090571
# category: ct 0.06426385696977377
# 0:01:12.819556
# 6.014488281682134
# 0:00:31.262922
# category: interest2 0.7753136167302728
# 0:02:28.482224
# category: interest3 0.1285447021946311
# 0:00:41.729535
# category: interest4 0.12854458391666412
# 0:00:38.047018
# 6.321893344633281
# 0:00:30.675995
# 5.331706457771361
# 0:00:37.852279
# 5.385227124206722
# 0:00:37.810654
# category: kw3 0.39494442008435726
# 0:00:52.155713
# category: marriageStatus 0.06426044460386038
# 0:00:39.479221
# category: os 0.06425812374800444
# 0:00:38.517371
# 4.89992605894804
# 0:00:33.520853
# 5.046098406426609
# 0:00:31.980332
# category: topic3 0.46153272688388824
# 0:00:50.930926

# aid                   int16
# uid                   int32
# label                 int8
# advertiserId          int32
# campaignId            int32
# creativeId            int32
# creativeSize          int8
# adCategoryId          int16
# productId             int32
# productType           int8
# LBS                   float32
# age                   int8
# appIdAction           category
# appIdInstall          category
# carrier               int8
# consumptionAbility    int8
# ct                    category
# education             int8
# gender                float32
# house                 float32
# interest1             object
# interest2             category
# interest3             category
# interest4             category
# interest5             object
# kw1                   object
# kw2                   object
# kw3                   category
# marriageStatus        category
# os                    category
# topic1                object
# topic2                object
# topic3                category
# dtypes: category(10), float32(3), int16(2), int32(5), int8(7), object(6)
# memory usage: 39.1 GB
#71.3G下降到65.3G,通过改变Object类型，再降到39.1G
#固定特征数据类型读取速度
import pandas as pd
from datetime import datetime
t1=datetime.now()
column_types={ 	
				'aid':'int16',
			  	'uid':'int32',
			  	'label':'int8',
			  	'advertiserId':'int32',
				'campaignId':'int32',
				'creativeId':'int32',
				'creativeSize':'int8',
				'adCategoryId':'int16',
				'productId':'int32',
				'productType':'int8',
				'age':'int8',
				'carrier':'int8',
				'consumptionAbility':'int8',
				'education':'int8',
				'LBS':'float32',
				'gender':'float32',
				'house':'float32',
				'appIdAction':'category',
				'appIdInstall':'category',
				'ct':'category',
				'interest2':'category',
				'interest3':'category',
				'interest4':'category',
				'kw3':'category',
				'marriageStatus':'category',
				'os':'category',
				'topic3':'category'
				}
train_test1_2_data=pd.read_csv("train_test1_2_data.csv",dtype=column_types)
t2=datetime.now()
print(t2-t1)#修改int和float类型时，从10:13s的时间下降到8:51s
#加上部分category特征，从8:51s上升到11:22s，但从71.3G下降到39.1G
train_test1_2_data.info(memory_usage='deep')

#一共68996077 训练45539700 测试23456377 测试1：11729073 测试2：11727304
train=train_test1_2_data[train_test1_2_data.label!=-1]#训练集 45539700
#aid的平均转换率
transfer_num=(train.label==1).sum()#2182403次转换
print(transfer_num/len(train))#0.04792 和官网说的正负样本1:20接近
# 832个广告
# 1993    903216
# 1125    705993
# 617     644241
# 651     626597
# 1410    588355
# 1039    579089
# 1438    565995
# 1474    541540
# 2195    512153
# 1711    496478
# 1364    461922
# 1974    403455
# 1690    387561
# 1937    379792
# 1180    376113
# 1984    370810
# 1022    366566
# 1244    363248
# 25      359586
# 667     333730
# 2179    332213
# 1139    323429
# 1256    322111
# 2028    313342
# 849     310127
# 855     299215
# 1502    295644
# 1081    293887
# 1571    286982
# 1700    283671
#          ...  
# 1791      7199
# 1836      7193
# 706       7192
# 2096      7184
# 1568      7184
# 1411      7181
# 968       7167
# 23        7086
# 1466      7063
# 1028      7063
# 1959      7050
# 569       7018
# 365       7000
# 1253      6966
# 895       6947
# 658       6934
# 408       6899
# 740       6859
# 2203      6782
# 456       6773
# 229       6765
# 677       6672
# 1150      6620
# 732       6612
# 1482      6524
# 343       6493
# 293       6359
# 133       6347
# 1229      5743
# 723       5166








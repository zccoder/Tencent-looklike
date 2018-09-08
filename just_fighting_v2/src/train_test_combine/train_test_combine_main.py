import os
os.system("python userFeature_to_DataFrame_cs.py")#将"userFeature_cs.data"转换成csv文件格式，方便进行pandas操作
os.system("python userFeature_to_DataFrame_fs.py")#将"userFeature_fs.data"转换成csv文件格式，方便进行pandas操作
os.system("python concat_cs_fs.py")#将初赛和复赛的adFeature、userFeature、train进行合并
os.system("python all_label_encoder_feature.py")#对adFeature和userFeature进行labelencoder
os.system("python extract_all_primary_labelencoderfeature.py")#对adFeature和userFeature进行labelencoder的结果merge到train和test上，得到train和test的primary_labelencoder_x.csv文件
os.system("python get_train_test.py")#对adFeature和userFeature merge到train,test上，得到train_test_data.csv

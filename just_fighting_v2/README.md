1.代码环境配置和所需依赖库
Python代码
tensorflow-gpu 1.4.1
Keras 2.1.3
lightgbm 2.1.1
pandas 0.20.3
numpy 1.14.2
scikit-learn 0.19.0
2.结果产出步骤说明
2.1将初赛原始文件放入"data目录下",并按如下命名
userFeature_cs.data   初赛用户特征文件
userFeature_fs.data   复赛用户特征文件
adFeature_cs.csv      初赛广告特征文件
adFeature_fs.csv      复赛广告特征文件
train_cs.csv          初赛训练集
train_fs.csv          复赛训练集
test.csv              测试集
2.2运行顺序
运行"bash run.sh"即可，具体运行顺序如下
(1)首先运行src/train_test_combine/train_test_combine_main.py,它会运行多个子文件，主要功能是首先将"userFeature_cs.data"和"userFeature_cs.data"转换成csv文件格式,方便进行pandas操作,接着将初赛和复赛的adFeature、userFeature、train进行合并，充分利用初赛和复赛的训练集，然后对adFeature和userFeature进行labelencoder，对adFeature和userFeature labelencoder的结果merge到train和test上，得到train和test的primary_labelencoder_x.csv文件，后续lightgbm会用到，直接作为特征放入，最后对adFeature和userFeature merge到train,test上，得到train_test_data.csv。具体内容查看src/train_test_combine/train_test_combine_main.py及其它们的子文件。
(2)接着运行src/uid_pos_neg_feature/uid_pos_neg_main.py,它会运行多个子文件，主要功能是提取用户的正负aid、正负creativeSize、正负productId、正负productType、正负adCategoryId，具体的比如用户的正负aid，当label为1的时候，训练集中某用户浏览的aid记录保存为list格式，即为该用户的正aid,当label=0时，训练集中该用户浏览的aid记录保存为list格式，即为该用户的负aid,为了防止利用未来数据，该用户的正负aid需要删去当前的记录的aid情况，这些特征后续的cv_NN和embedding_NN中需要使用。具体内容查看src/uid_pos_neg_feature/uid_pos_neg_main.py及其它们的子文件。
(3)运行src/cv_feature/cv_main.py及其它们的子文件，主要功能是对原始的特征和用户的正负aid进行countvector和用户的正负creativeSize、正负productId、正负productType、正负adCategoryId进行tfidf，然后将得到的结果文件作为稀疏矩阵的特征给入cv_NN中。具体内容查看src/cv_feature/cv_main.py及其它们的子文件。
(4)运行src/transfer_feature/transfer_feature/transfer_main.py,它会运行多个子文件，主要功能是对多个特征进行交叉合并，生成新的特征，将结果保存下来，方便后续的使用，其余特征有多值的长度特征和转换率特征，对于多值的转换率特征是求出每个单值的转换率特征后进行平均，为了防止使用未来的数据，此处使用五折交叉的方法求得所有的转换率，此外，由于某些特征出现的次数比较少，所以用了平滑，减少影响。由于复赛数据量大，所以只选取初赛lightgbm按重要性从高到低排序后的前100多个，以及分成了多个文件提取特征，一方面为了防止内存溢出，二是可以更方便的比较新加入的特征对结果的影响大小。
(5)运行src/cv_NN/countvector_NN.py,它使用了第(3)中提到的cv特征以及(4)中提到的部分转换率特征，将它们合并后按稀疏矩阵的形式输入到三层神经网络中，迭代两次，并bagging多次,运行得到的结果保存到submission_result目录中，模型保存到models中。
(6)运行src/LGB/lgb_transfer_feature.py,主要功能是使用lightgbm模型预测结果，特征包括原始特征，即labelencoder特征和(4)中的所有转换率特征，运行得到的结果保存到submission_result目录中，模型保存到models中。
(7)embedding_feature
(8)embedding_NN
(9)最后运行src/submission_result/vote.py,对embedding_NN,cv_NN和lightgbm进行加权融合，生成最终结果文件保存到src/submission.csv中。
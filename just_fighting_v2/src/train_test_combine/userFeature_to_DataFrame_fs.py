import pandas as pd
userFeature_data=[]
print("fs userFeature transform csv file start:")
input_path='../../data/'
with open(input_path+'userFeature_fs.data','r') as f:
	for i,line in enumerate(f):
		line=line.strip().split('|')#按'|'切分，切分后的每个内容为一个特征
		userFeature_dict={}
		for each in line:#遍历所有特征
			each_list=each.split(' ')#按空格分隔，each_list[0]为列名,[1-n]为特征值
			userFeature_dict[each_list[0]]=' '.join(each_list[1:])#对[1-n]进行合并，空格分隔
		userFeature_data.append(userFeature_dict)
		if i%10000000==0:
			print(i)
	print("total data:",i+1)
	user_feature=pd.DataFrame(userFeature_data)
	user_feature.to_csv(input_path+"userFeature_fs.csv",index=False)
print("fs userFeature transform file end")
#uid 26325489|age 4|gender 2|marriageStatus 11|education 7|consumptionAbility 2|LBS 950|interest1 93 70 77 86 109 47 75 69 45 8 29 49 83 6 46 36 11 44 30 118 76 48 28 106 59 67 41 114 111 71 9|interest2 46 19 13 29|interest5 52 100 72 131 116 11 71 12 8 113 28 73 6 132 99 76 46 62 121 59 129 21 93|kw1 664359 276966 734911 103617 562294|kw2 11395 79112 115065 77033 36176|topic1 9826 105 8525 5488 7281|topic2 9708 5553 6745 7477 7150|ct 3 1|os 2|carrier 1
#uid 1184123|age 2|gender 1|marriageStatus 5 13|education 2|consumptionAbility 1|LBS 803|interest1 75 29|interest2 33|kw1 338851 361151 542834 496283 229952|kw2 80263 39618 53539 180 38163|topic1 4391 9140 5669 1348 4388|topic2 9401 7724 1380 8890 7153|ct 3 1|os 1|carrier 1
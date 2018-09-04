'''
Created on 2018年4月22日

@author: Jiao
'''
import pandas as pd

AD_FEATURE_FILE = '../data/adFeature.csv'
USER_FEATURE_FILE = '../data/userFeature.data'
TRAIN_FILE = '../data/train.csv'
TEST_FILE = '../data/test.csv'
TRAIN_COMBINE_CSV = '../data/train_combine.csv'
TEST_COMBINE_CSV = '../data/test_combine.csv'
all_user_group_names = ['age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS', 
                        'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                        'kw1', 'kw2', 'kw3',
                        'topic1', 'topic2', 'topic3',
                        'appIdInstall',
                        'appIdAction',
                        'ct',
                        'os',
                        'carrier',
                        'house']

ad_feature=pd.read_csv(AD_FEATURE_FILE, dtype='str')
ad_headers = list(ad_feature)

ad_dict = {}
for i in range(0, len(ad_feature)):
    row = ad_feature.loc[i]
    ad_dict[row[0]] = ','.join(row[1:])

user_tmpl = []
user_idx = {}
for i, fname in enumerate(all_user_group_names):
    user_tmpl.append('0')       #default '0'
    user_idx[fname] = i
     
user_dict = {}
with open(USER_FEATURE_FILE, 'r') as f:
    for i, line in enumerate(f):
        line = line.strip().split('|')
        uid = line[0].split(' ')[1]
        header_idx = 0
        user_data = user_tmpl.copy()
        for each in line[1:]:
            line_list = each.split(' ')
            user_data[user_idx[line_list[0]]] = ' '.join(line_list[1:])     #lookup for key position
        user_dict[uid] = ','.join(user_data)
        if i % 500000 == 0 and i > 0:
            print(i)
            print(user_dict[uid])

user_none_data = ','.join(user_tmpl)   
ad_none_data = ','.join(['0' for x in range(len(ad_headers) - 1)])
def merge_user_feature_ad_feature(fileInput, fileOut,add_label=False):
    fi = open(fileInput, 'r')
    fo = open(fileOut, 'w')
    #headers
    line = fi.readline().strip()
    if add_label:
        line = line + ',label,' + ','.join(all_user_group_names) + ',' + ','.join(ad_headers[1:]) + '\n'
    else:
        line = line + ',' + ','.join(all_user_group_names) + ',' + ','.join(ad_headers[1:]) + '\n'
    fo.write(line)
    line = fi.readline()
    while line:
        line =  line.strip()
        line_list = line.split(',')
        aid = line_list[0]
        uid = line_list[1]
        if add_label:
            line = line + ',-1'
        if uid in user_dict:
            user_data = user_dict[uid]
        else:
            user_data = user_none_data
        if aid in ad_dict:
            ad_data = ad_dict[aid]
        else:
            ad_data = ad_none_data
        line = ','.join([line, user_data, ad_data]) + '\n'
        fo.write(line)
        line = fi.readline()
    fi.close()
    fo.close()
print('Bulding Train File...')
merge_user_feature_ad_feature(TRAIN_FILE, TRAIN_COMBINE_CSV)
print('Bulding Test File...')
merge_user_feature_ad_feature(TEST_FILE, TEST_COMBINE_CSV, True)
# -*- encoding:utf:8 -*-

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import gc
import warnings
warnings.filterwarnings("ignore")


def remove_lowcase(se):
    count = dict(se.value_counts())
    se = se.map(lambda x : -1 if count[x]<10 else x)
    return se


def static_feat(df,df_val, feature):
    df['label'] = df['label'].replace(-1,0)

    df = df.groupby(feature)['label'].agg(['sum','count']).reset_index()

    new_feat_name = feature + '_stas'
    df.loc[:,new_feat_name] = 100 * (df['sum'] + 1) / (df['count'] + np.sum(df['sum']))
    df.loc[:,new_feat_name] = np.round(df.loc[:,new_feat_name].values,4)
    # print(df.head())
    df_stas = df[[feature,new_feat_name]]
    df_val = pd.merge(df_val, df_stas, how='left', on=feature)

    return df_val


def main():
    # reader = pd.read_csv('./data/commbine_val.csv', header=0, iterator=True)
    chunk_size = 200000
    ad_feat_list = ['advertiserId','campaignId','creativeSize','adCategoryId','productType','productId']
    user_feat_list = ['LBS','age','consumptionAbility','education','gender','os','ct','marriageStatus','house','carrier']
    df_features = None
    i = 0
    for afeat in ad_feat_list:
        for ufeat in user_feat_list:
            concat_feat = afeat + '_' + ufeat
            # if concat_feat in ['creativeSize_LBS','adCategoryId_education','creativeSize_age','productId_LBS','advertiserId_gender']:
            #     continue

            reader = pd.read_csv('./data/commbine_val.csv', header=0, iterator=True)
            chunks = []
            loop = True
            while loop:
                try:
                    chunk = reader.get_chunk(chunk_size)[['aid',afeat, ufeat,'label']]
                    chunks.append(chunk)
                except StopIteration:
                    loop = False
                    print("Iteration is stopped")

            df = pd.concat(chunks,axis=0, ignore_index=True)
            del chunks
            gc.collect()

            df[concat_feat] = df['aid'].astype('str') + '_' + df[afeat].astype('str') + '_' +df[ufeat].astype('str')
            df = df[[concat_feat,'label']]
            df['index'] = list(range(df.shape[0]))
            df[concat_feat] = remove_lowcase(df[concat_feat])

            if i == 0:
                df[['index','label']].to_csv('./data/static-feat/label.csv',index=False)
            # df[ufeat] = df[ufeat].map(lambda x : ' '.join(sorted(str(x).split())))
            i += 1
            
            df_stas_feat = None
            kf = KFold(n_splits = 5,random_state=2018,shuffle=True)
            for train_index, val_index in kf.split(df):
                X_train = df.loc[train_index,:]
                X_val = df.loc[val_index,:]

                X_val = static_feat(X_train,X_val, concat_feat)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)


            del df_stas_feat['label']
            del df_stas_feat[concat_feat]
            # del df['label']
            # df = pd.merge(df,df_stas_feat, how='left',on=concat_feat)
            # print(df_stas_feat.shape,df_stas_feat.head())

            # df_features = pd.concat([df_features,df],axis=1, ignore_index=True)
            df_stas_feat.to_csv('./data/static-feat/%s-%s.csv' % (afeat, ufeat),index=False)
            print('save succeed')
            del df
            gc.collect()

            print(afeat, ufeat,'done!')
            

main()












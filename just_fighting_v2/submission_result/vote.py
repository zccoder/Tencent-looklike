import pandas as pd
a=pd.read_csv("embedding_NN.csv")
b1=pd.read_csv("cv_NN1.csv")
b2=pd.read_csv("cv_NN2.csv")
c=pd.read_csv("submission_LGB.csv")

res=a[['aid','uid']]
res['score']=a['score']*0.4+((b1['score']+b2['score'])/2)*0.4+c['score']*0.2
res['score']=res['score'].apply(lambda x:float('%.6f'%x))
res.to_csv('../submission.csv',index=None)
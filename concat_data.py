"""
This file is to concat liveroom data and y_train in chronological order
NB: live room data 有毫秒的数据，需要合并到整秒
"""
import pandas as pd
import numpy as np

df1 = pd.read_csv('./liveroom_data.csv')
df2 = pd.read_csv('./data/y_train.csv')

df2.drop_duplicates(subset='time', inplace=True)

df2.columns = ['a','index','time','result', 'y']

m = df2['time'].values
d = m.copy()
for i in range(len(m)):
    if '2019 - 08 - 19 20:01:39' <= m[i] <= '2019 - 08 - 19 23:59:59':
        d[i] = m[i][0:4]+m[i][5:6]+m[i][7:9]+m[i][10:11]+m[i][12:]
df2['time'] = d

time_list = df2.time.tolist()

df = pd.DataFrame(columns=['time', 'enterroom_lv', 'gift', 'online_num', 'pk', 'rank_region',
       'sysbilibili', 'thumbs'])
for i in range(len(time_list)-1):
    df_temp = df1[(df1['time']>time_list[i]) & (df1['time']<=time_list[i+1])]
    temp = df_temp[['enterroom_lv', 'gift', 'online_num', 'pk', 'rank_region',
       'sysbilibili', 'thumbs']].max()
    temp['time'] = time_list[i+1]
    df = df.append(temp,ignore_index=True)

df.to_csv('data/input_merge.csv')


df_train = pd.merge(df,df2, how='left',on=['time'])
df_train.drop(['a','index'],axis=1,inplace=True)

df_train.to_csv('./tmp_restore/train_set.csv')

df = pd.read_csv('./tmp_restore/train_set.csv')

df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M:%S')

df[['online_num','thumbs']] = df[['online_num','thumbs']].replace(0.0, np.nan)

df.loc[0,['thumbs']] = 0

df[['online_num','thumbs']] = df[['online_num','thumbs']].interpolate()

df['online_num'] = df['online_num'].astype('int')
df['thumbs'] = df['thumbs'].apply(lambda x: x//10*10).astype('int')
df.to_csv('./result_data/train_data.csv')

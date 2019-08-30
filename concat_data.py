"""
This file is to concat liveroom data and y_train in chronological order
NB: live room data 有毫秒的数据，需要合并到整秒
"""

import pandas as pd
import json

# 请将这段送至anaconda notebook执行。pycharm 报错，推测是pandas版本问题
# path = '/Users/momo/VertualActor/liveroom_data.json'
# df = pd.read_json(path, lines=True)
# df['time'] =pd.to_datetime(df.time)
# df.sort_values('time', inplace=True)
# df.to_csv('/Users/momo/VertualActor/liveroom_data.csv')


def f(r):
    r['time'] = r['time'].split('.')[0]
    return r

path = '/Users/momo/VertualActor/liveroom_data.csv'

df = pd.read_csv(path)
new_df = df.apply(f, axis=1).groupby('time').max()  # axis=1表示apply到列


"""
This file is to filter live room raw data to regular row/col
"""

import pandas as pd
import json
import os

# 合并单日的直播间数据为一个txt
def mergetxt():
    path = './data/liveroom_data/'
    with open('./data/liveroomdata.txt', 'w') as f:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    for line in open(file_path, encoding='UTF-8'):
                        f.write(line)
                    f.write('\n')

# 建立dict of gift-price
def to_giftdict(gift_path):
    dict_gift = {}

    for line in open(gift_path):
        if line.find('int') is not -1:
            if len(line) > 16:
                id = str(line.split('int(')[1].split(')')[0])
                dict_gift[id] = ''
            else:
                unit_price = (line.split('int(')[1].split(')')[0])
                dict_gift[id] = unit_price

    return dict_gift


# 处理小时榜、地区榜
def process_rank(df):
    df['rank_hour'] = 0
    df['region'] = 0

    for index, row in df.iterrows():
        if row['rank'].find('小时榜') is not -1:
            df.loc[index, 'rank_hour'] = row['rank'].split('第')[1].split('名')[0]
        if row['rank'].find('湖南') is not -1:
            try:
                df.loc[index, 'region'] = row['rank'].split('第')[1].split('名')[0]
            except IndexError:
                print(index)
        if row['rank_region'].find('湖南') is not -1:
            try:
                df.loc[index, 'region'] = row['rank_region'].split('第')[1].split('名')[0]
            except IndexError:
                print(index)

    return df


# 1 load raw data
path = './data/liveroomdata.txt'
file = open(path, 'rb')

js = file.read().decode('utf-8')
type(js)


# 2 load gift-price table
gift_path = './data/gift.txt'
dict_gift = to_giftdict(gift_path)
exist_giftid = dict_gift.keys()

# 3 filter raw data & write to new json
dict = {}
msgType = ''
msgUnit = ''
with open('./liveroom_data.json', 'w') as f:

    for line in open(path, encoding='UTF-8'):

        if len(line) < 300:  # filter error data
            continue
        if line.find('ERROR') is not -1:
            continue
        if line.find('Error') is not -1:
            continue
        if line.find('"activityBarType":null') is not -1:
            continue

        dict['online_num'] = 0    # 在线人数
        dict['rank_region'] = 0   # 地区榜单2
        dict['gift'] = 0          # 礼物价格
        dict['thumbs'] = 0        # 星光值
        dict['sysbilibili'] = 0   # 飘屏
        dict['enterroom_lv'] = 0  # 进场大佬的财富等级
        dict['pk'] = 0            # 判定是否为pk模式；1/0
        dict['rank'] = 0          # 榜单1

        time_str = line.split('],body:[')[0].split('time:[')[1]
        dict['time'] = time_str.split('+')[0]

        body = line.split(',body:[')[1].split(']\n')[0]
        if body == "":
            continue
        body = json.loads(body)                 # 读取每一行，将每一行读取成为json文件
        # print(body)

        pk_judge = '"activityBarType":"PK"'   # 判断是否为PK模式
        if line.find(pk_judge) is not -1:
            dict['pk'] = 1
            # print(line)

            j = json.dumps(dict)
            f.write(j)
            f.write('\n')
            continue

        if line.find('activityMsgType":"GROUP') is not -1:  #过滤未知数据行
            continue

        try:
            msgUnit = body['msgUnit']
            msgType = msgUnit['type']
        except TypeError:
            print(line)

        if msgType == 'BillboardBar':
            dict['rank_region'] = msgUnit['BillboardBar.data']['rankText']
        elif msgType == 'Gift':
            buytimes = int(msgUnit['Pay.Gift.data']['buytimes'])
            gift_id = msgUnit['Pay.Gift.data']['product_id']
            if gift_id in exist_giftid:
                unit_price = dict_gift[gift_id]
            else:
                unit_price = 1            # 处理gift 表里缺省的id, 单价置为1
            dict['gift'] = buytimes*int(unit_price)
        elif msgType == 'LevelChange':
            dict['rank'] = msgUnit['LevelChange.data']['rankText']
        elif msgType == 'Thumbs':
            dict['thumbs'] = msgUnit['Pay.Thumbs.data']['thumbs']
        elif msgType == 'SysBiliBili':
            dict['sysbilibili'] = msgUnit['Pay.SysBiliBili.data']['text']
        elif msgType == 'EnterRoom':
            try:
                dict['enterroom_lv'] = msgUnit['Set.EnterRoom.data']['fortune_lv']
            except TypeError:
                print('enteroom_lv error:', line)
        elif msgType == 'RoomOnlineNum':
            dict['online_num'] = msgUnit['Set.RoomOnlineNum.data']['online_number']
        else:
            continue

        j = json.dumps(dict, ensure_ascii=False)

        f.write(j)
        f.write('\n')

"""
合并毫秒数据为整秒
"""
def f(r):
    r['time'] = r['time'].split('.')[0]
    return r

path = './liveroom_data.json'
df = pd.read_json(path, lines=True)
df['time'] = pd.to_datetime(df.time)
df.sort_values('time', inplace=True)
df.to_csv('./tmp_restore/tmp_data.csv', index=None)      # 重新read一次，否则time型数据无法split
df = pd.read_csv('./tmp_restore/tmp_data.csv')
new_df = df.apply(f, axis=1).groupby('time').max()   # axis=1表示apply到列
new_df = process_rank(new_df)
new_df.to_csv('./liveroom_data.csv')
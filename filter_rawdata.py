"""
This file is to filter live room raw data to regular row/col
"""

import pandas as pd
import json

# 1 load raw data
path = './data/live_rawdata.txt'
file = open(path, 'rb')

js = file.read().decode('utf-8')
type(js)

dict = {}
msgType = ''
nsgUnit = ''

# 2 load gift-price table
gift_path = './data/gift.txt'
dict_gift = {}

for line in open(gift_path):
    dict_gift['2'] = 1     # product 为2的找不到价格
    dict_gift['1'] = 1
    if line.find('int') is not -1:
        if len(line) > 16:
            id = str(line.split('int(')[1].split(')')[0])
            dict_gift[id] = ''
        else:
            unit_price = (line.split('int(')[1].split(')')[0])
            dict_gift[id] = unit_price

# 3 filter raw data & write to new json
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
        dict['rank_region'] = 0   # 地区榜
        dict['gift'] = 0          # 礼物价格
        dict['thumbs'] = 0        # 星光值
        dict['sysbilibili'] = 0   # 飘屏
        dict['enterroom_lv'] = 0  # 进场大佬的财富等级
        dict['pk'] = 0            # 判定是否为pk模式；1/0
        dict['rank_hour'] = 0     # 小时榜

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

        msgUnit = body['msgUnit']
        msgType = msgUnit['type']


        if msgType == 'BillboardBar':
            dict['rank_region'] = msgUnit['BillboardBar.data']['rankText']
        elif msgType == 'Gift':
            buytimes = int(msgUnit['Pay.Gift.data']['buytimes'])
            gift_id = msgUnit['Pay.Gift.data']['product_id']
            unit_price = dict_gift[gift_id]
            dict['gift'] = buytimes*int(unit_price)
        elif msgType == 'LevelChange':
            dict['rank_hour'] = msgUnit['LevelChange.data']['rankText']
        elif msgType == 'Thumbs':
            dict['thumbs'] = msgUnit['Pay.Thumbs.data']['thumbs']
        elif msgType == 'SysBiliBili':
            dict['sysbilibili'] = msgUnit['Pay.SysBiliBili.data']['text']
        elif msgType == 'EnterRoom':
            dict['enterroom_lv'] = msgUnit['Set.EnterRoom.data']['fortune_lv']
        elif msgType == 'RoomOnlineNum':
            dict['online_num'] = msgUnit['Set.RoomOnlineNum.data']['online_number']
        else:
            continue

        j = json.dumps(dict, ensure_ascii=False)

        f.write(j)
        f.write('\n')


# path = '/Users/momo/Company/script_python/liveroom_data.json'
# file = open(path, 'rb')
#
# js = file.read().decode('utf-8')
# type(js)
#
# df_empty = pd.DataFrame()
#
# for line in open(path, encoding='UTF-8'):
#     data_list = json.loads(line)                 # 读取每一行，将每一行读取成为json文件
#     data_df = pd.DataFrame(data_list, index=[0]) # 将每一行转成data frame的形式
#     df_empty = df_empty.append(data_df)
#
# print(df_empty.head())
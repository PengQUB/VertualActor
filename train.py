# pip install xgboost==0.80
# 绘图需要pip install graphviz
# mac 还需要 brew install graphviz

import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        if feat.find('Unnamed') == -1:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1

    outfile.close()


def down_sample(df):
    df1 = df[df['label'] != 0]#正例
    df0 = df[df['label'] == 0]##负例
    df3=df0.sample(frac=0.01)##抽负例
    return pd.concat([df1, df3], ignore_index=True)


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softmax'  # 多分类、输出概率值
    param['eta'] = 0.1  # lr
    param['max_depth'] = 6  # max depth
    param['silent'] = 1  # train info
    param['num_class'] = 6
    param['min_child_weight'] = 1  # 停止条件，这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    param['subsample'] = 0.7  # 随机采样训练样本
    param['colsample_bytree'] = 0.7   # 生成树时进行的列采样
    param['seed'] = seed_val
    num_rounds = num_rounds  # 迭代次数

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    pred_train_y = model.predict(xgtrain)
    return pred_test_y, pred_train_y, model


if __name__ == '__main__':

    # 1 导入数据
    data_path = "./result_data/"
    train_file = data_path + "train_data_21_22.csv"
    train_df = pd.read_csv(train_file)
    train_df_total = train_df.fillna(0)
    train_df = down_sample(train_df_total)      # 样本不均衡，故对负例用欠采样
    print('after dowmsample', train_df.shape)


    # 2 构建特征
    # We do not need any pre-processing for numerical features and so create a list with those features.
    features_to_use = ["enterroom_lv", "gift", "online_num", "thumbs", "rank_hour", "region", "time_sum"]

    # Now let us create some new features from the given features.
    train_df["sysbilibili"] = train_df["sysbilibili"].apply(len)
    train_df_total["sysbilibili"] = train_df_total["sysbilibili"].apply(len)


    # convert the created column to datetime object so as to extract more features
    train_df["time"] = pd.to_datetime(train_df["time"])

    # Let us extract some features like year, month, day, hour from date columns #
    train_df["time_year"] = train_df["time"].dt.year
    train_df["time_month"] = train_df["time"].dt.month
    train_df["time_day"] = train_df["time"].dt.day
    train_df["time_hour"] = train_df["time"].dt.hour


    # adding all these new features to use list #
    features_to_use.extend(["time_year", "time_month", "time_day", "time_hour", "sysbilibili"])

    features = list(train_df[features_to_use])  # create xgb.fmap for plot
    ceate_feature_map(features)


    train_X = np.array(train_df[features_to_use])
    train_y = np.array(train_df['label']).astype(int)


    acc = []
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016) # random_state=2016表示下次取出的样本还是相同的
    for dev_index, val_index in kf.split(range(train_X.shape[0])):
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        print(dev_X.shape)
        print(val_X.shape)
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, preds_train, model = runXGB(dev_X, dev_y, val_X, val_y)  # model training

        acc.append(accuracy_score(val_y, preds))

        print(preds)
        print('acc', acc)
        print(val_y)
        break

    xgtrain = xgb.DMatrix(train_X, label=train_y)
    total_preds = model.predict(xgtrain)
    from sklearn import metrics
    target_names = ['0', '1', '2', '3', '4', '5']
    print(metrics.classification_report(val_y, preds, target_names=target_names))
    print(metrics.classification_report(train_y, total_preds, target_names=target_names))

    # print(metrics.classification_report(train_y_total_y, total_preds, target_names=target_names))  # 总样本集上的metric


    plot_tree(model, fmap='xgb.fmap')
    fig = plt.gcf()
    fig.set_size_inches(150, 100)
    fig.savefig('./tree.png')

    # 拼接val_X和preds
    # df1 = pd.DataFrame(val_X)
    # df2 = pd.DataFrame(preds)
    # df = pd.concat([df1, df2], axis=1)
    # df.to_csv('./prediction.csv')
import pandas as pd

path = './yege_out_19_27.csv'

df = pd.read_csv(path, header=0, names=["i", "j", "phrase", "label", "time", "source"])

df = df[df.time > "2019-08-20 23:59:59"]
df.sort_values('time', inplace=True)
df.to_csv('./result.csv', index=None)
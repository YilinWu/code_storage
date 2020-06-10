from com_uitl import *

import pandas as pd
import time
import numpy as np
import psutil
import os
import gc
import lightgbm as lgb
import pytz


path = '/home/cczaixian/Documents/TalkingData_AdTracking_Fraud_Detection_Challenge/data/'
process = psutil.Process(os.getpid())
print('Total memory in use before reading train: ', process.memory_info().rss/(2**30), ' GB\n')    

#######  READ THE DATA  #######

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'click_id'      : 'uint32'
        }
print('\nLoading train...')
train_cols = ['ip','app','device','os', 'channel', 'click_time','is_attributed']
test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
train = pd.read_csv(path+"train.csv", dtype=dtypes,usecols=train_cols, parse_dates=['click_time'])
test = pd.read_csv(path+"test_supplement.csv", parse_dates=['click_time'],dtype=dtypes, usecols=test_cols)

df_all = train.append(test)

def click_time(df):
    cst = pytz.timezone('Asia/Shanghai')
    df['click_time'] = pd.to_datetime(df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
    df['day'] = df.click_time.dt.day.astype('uint8')
    df['hour'] = df.click_time.dt.hour.astype('uint8')
    return df

df_all = click_time(df_all)



def df_add_counts(df, cols, tag="_count"):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols)+tag] = counts[unqtags]
    return df

def df_add_uniques(df, cols, tag="_unique"):
    gp = df[cols].groupby(by=cols[0:len(cols) - 1])[cols[len(cols) - 1]].nunique().reset_index(). \
        rename(index=str, columns={cols[len(cols) - 1]: "_".join(cols)+tag})
    df= df.merge(gp, on=cols[0:len(cols) - 1], how='left')
    return df

df_all=df_add_counts(df_all, ['ip'])
df_all=df_add_counts(df_all, ['ip', 'app'])
df_all=df_add_counts(df_all, ['ip', 'app', 'os'])

df_all=df_add_counts(df_all, ['ip', 'device'])
df_all=df_add_counts(df_all, ['app', 'channel'])

df_all=df_add_counts(df_all, ['ip', 'day', 'hour'])
df_all=df_add_counts(df_all, ['ip', 'os', 'day', 'hour'])
df_all=df_add_counts(df_all, ['ip', 'app', 'day', 'hour'])
df_all=df_add_counts(df_all, ['ip', 'device', 'day', 'hour'])
df_all=df_add_counts(df_all, ['day', 'hour', 'app'])

df_all = df_add_uniques(df_all, ['ip', 'channel'])


train_count = df_all[df_all.is_attributed.notnull()]
train_count.drop(train_cols, axis=1, inplace=True)
train_count.drop(['click_id'], axis=1, inplace=True)
#print(train_count)
train_count.to_pickle('data/feature_count_train.pkl')



test_count = df_all[df_all.click_id.notnull()]
test_count.drop(test_cols, axis=1, inplace=True)
test_count.drop(['is_attributed'], axis=1, inplace=True)
#print(test_count)
train_count.to_pickle('data/feature_count_train.pkl')

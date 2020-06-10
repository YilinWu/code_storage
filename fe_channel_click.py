from com_uitl import *

import pandas as pd
import time
import numpy as np
import psutil
import os
import gc
import lightgbm as lgb
import pytz

from time import time



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

train_cols = [ 'channel', 'click_time','is_attributed']
test_cols = ['channel', 'click_time', 'click_id']
train = pd.read_csv(path+"train.csv", dtype=dtypes,usecols=train_cols, parse_dates=['click_time'])
test = pd.read_csv(path+"test_supplement.csv", parse_dates=['click_time'],dtype=dtypes, usecols=test_cols)

#change click_time 
ONE_SECOND = 1000000000
train['click_time'] = train.click_time.astype('int64').floordiv(ONE_SECOND).astype('int32')
test['click_time'] = test.click_time.astype('int64').floordiv(ONE_SECOND).astype('int32')

#set a index to join back
train['index_col'] = train.index
test['index_col'] = test.index



#base on ip ,device ,os 
def cal_recent_click_cnt(train_df, recent_time_range, key='recent_click_cnt'):# recent_time_range is in seconds
    train_df = train_df[['index_col','channel', 'click_time']] 
    train_v = train_df.values.tolist()
    tot_v = train_v 
    tot_v = sorted(tot_v, key=lambda d: d[1]) #sorted by col 2

    feat = []
    default_v = -1
    has_feat = 0
    for i, rec in enumerate(tot_v): # count 
        if i == 0:
            one = rec[:2] + [default_v, ]   #first row 'index_col','channel',-1

        elif rec[1] == tot_v[i - 1][1]: #if same 'channel',-1as former one
            cnt = 0                         #start from 0
            j = i - 1                       # set j as i-1
            while tot_v[j][1] == rec[1] and rec[2] - tot_v[j][2] <= recent_time_range: #if click_time gap in time range count +1
                cnt += 1
                j -= 1  # j-1
            one = rec[:2] + [cnt]           #second 'index_col','channel', 1
            if cnt > 0:
                has_feat += 1
        else:
            one = rec[:2] + [0, ]           #reset as 'index_col','channel', 0

        feat.append(one)

    key = key + '_' + str(recent_time_range) if key is not None else 'click_cnt_up_to_now'

    print ("cal:", key, " has_feat:", has_feat, "tot_rec:", len(tot_v))

    feat_df = pd.DataFrame(feat, columns=['index_col','channel', key])

    return feat_df[['index_col', key]]

#reset index
def merge(df1,df2,col):
    df2 = pd.merge(df1, df2, on=['index_col'])
    df2.drop(col, axis=1, inplace=True)
    df2.drop(['index_col'],axis=1, inplace=True)
    return df2


t0 = time()
print('start 2m')
train_2m = cal_recent_click_cnt(train, 180 ,key='recent_click_cnt')
train_2m=merge(train,train_2m,train_cols)
print('took %0.3fs' % (time() - t0))

t0 = time()
print('start 10')
train_10m =cal_recent_click_cnt(train, 600 ,key='recent_click_cnt')
train_10m=merge(train,train_10m,train_cols)
print('took %0.3fs' % (time() - t0))

t0 = time()
print('start 15')
train_15m =cal_recent_click_cnt(train, 900 ,key='recent_click_cnt')
train_15m=merge(train,train_15m,train_cols)
print('took %0.3fs' % (time() - t0))

t0 = time()
print('start 30')
train_30m =cal_recent_click_cnt(train, 1800 ,key='recent_click_cnt')
train_30m=merge(train,train_30m,train_cols)
print('took %0.3fs' % (time() - t0))

t0 = time()
print('start 60')
train_60m =cal_recent_click_cnt(train, 3600 ,key='recent_click_cnt')
train_60m=merge(train,train_60m,train_cols)
print('took %0.3fs' % (time() - t0))

print('append to train')
train_3click_train = train_2m.append([train_10m, train_15m,train_30m,train_60m])

print('to file')
train_3click_train.to_pickle('data/feature_3click_train.pkl')



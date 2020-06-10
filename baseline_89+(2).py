#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import time
import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from datetime import date, timedelta
import itertools

#########################load data##############################
DEBUG = 1
SUB = 1
if DEBUG:
    cache_path = '../cache_hour_new/'
else:
    cache_path = '../cache_pub2/'
load = True
SMALL = True
data_path = '../input/'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time','attributed_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

most_freq_hours_in_test_data = [4,5,9,10,13,14]
least_freq_hours_in_test_data = [6, 11, 15]

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

if DEBUG:
    nrows = None
    train = pd.read_csv(data_path+ 'train_for_hour2.csv', dtype=dtypes,nrows = nrows,header=0, usecols=train_cols,parse_dates=["click_time","attributed_time"])

#train.click_time = train.click_time + timedelta(hours=8)
train['hour'] = train["click_time"].dt.hour.astype('uint8')
train['day'] = train["click_time"].dt.day.astype('uint8')
train['minute'] = train["click_time"].dt.minute.astype('uint8')
train['15min'] = 4* train['hour'] + train['minute'] // 15
train['3min'] = 20* train['hour'] + train['minute'] // 3
train['30min'] = 2* train['hour'] + train['minute'] // 30

train.reset_index(drop=True,inplace=True)


if DEBUG:
    third_day_start_index = train[train.click_time>='2017-11-9'].index.min()  #122070318
print(third_day_start_index)
print(len(train)-third_day_start_index)
gc.collect()

#########################useful function from piupiu##############################

def concat(L):
    result = None
    for l in L:
        print(l.columns.tolist())
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))        
        print("done")
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except:
                print(l.head())
    return result


def left_merge(data1,data2,on):
    if type(on) != list:
        on = [on]
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    columns = [f for f in data2.columns if f not in on]
    result = data1.merge(data2_temp,on=on,how='left')
    result = result[columns]
    return result


def rank(data, feat1, feat2, ascending=True):
    data.sort_values(feat1 + feat2, inplace=True, ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1, as_index=False)['rank'].agg({'min_rank': 'min','max_rank':'max'})
    data = pd.merge(data, min_rank, on=feat1, how='left')
    data['rank'] = (data['rank'] - data['min_rank'])/(data['max_rank']-data['min_rank'])
    del data['min_rank'], data['max_rank']
    return data['rank']

######################### feature engerineng##############################


def get_feat_size(size_feat):
    """计算A组的数量大小（忽略NaN等价于count）"""
    result_path = cache_path +  ('_').join(size_feat)+'_count'+'.csv'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
        print("get_size_feat" + result_path)
        result = train[size_feat].groupby(by=size_feat).size().reset_index().rename(columns={0: ('_').join(size_feat)+'_count'})
        result = left_merge(train,result,on=size_feat)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_feat_size_feat(base_feat,other_feat):
    """计算唯一计数（等价于unique count）"""
    result_path = cache_path + ('_').join(base_feat)+'_count_'+('_').join(other_feat)+'.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
        print("get_size_feat_size" + result_path)
        result = train[base_feat].groupby(base_feat).size().reset_index()\
                      .groupby(other_feat).size().reset_index().rename(columns={0: ('_').join(base_feat)+'_count_'+('_').join(other_feat)})
        result = left_merge(train,result,on=other_feat)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result



def get_rank(base_feat,ascending=True):
    name = ('_').join(base_feat)+'_rank'+'_'+str(ascending)
    result_path = cache_path + ('_').join(base_feat)+'_rank'+'_'+str(ascending)+'.csv'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
        print("get rank"+ result_path)
        train[name] = train.groupby(base_feat).cumcount(ascending=ascending)
        result = train[[name]]
        train.drop([name],axis=1,inplace=True)      
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        gc.collect()
    return result

def get_next_click_time(base_feat,cat='next'):
    if cat=='next':
        shiftnum = -1
    elif cat=='last':
        shiftnum = 0
    name = ('_').join(base_feat)+'_next_click'+'_' + cat
    print(name)
    result_path = cache_path + ('_').join(base_feat)+'_next_click'+'_'+cat+'.csv'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
        train['category'] = train.groupby(base_feat).ngroup().astype('uint32')
        train.sort_values(['category','click_time'],inplace=True)
        train['catdiff'] = train.category.diff().shift(shiftnum).fillna(1).astype('uint8')
        train.drop(['category'],axis=1,inplace=True)
        train[name] = train.click_time.diff().shift(shiftnum).astype('int64').floordiv(1000000000).astype('float32')
        #train[name] = train[name].diff().shift(shiftnum)
        #train[name] = np.clip(train[name],0,10800) # train have only three hour a day
        train.loc[train.catdiff==1, name] = np.nan
        train.sort_index(inplace=True)
        result = train[[name]]
        train.drop(['catdiff',name],axis=1,inplace=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        gc.collect()
    return result



def get_feat_stat_feat(base_feat,other_feat,stat_list=['min','max','var','size','mean','skew']):
    result_path = cache_path + ('_').join(base_feat)+'_'+('_').join(stat_list)+'_'+('_').join(other_feat)+'.hdf'
    name = ('_').join(base_feat) +'_'+('_').join(other_feat)+'_'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
        print("get_stat" + result_path)
        agg_dict = {}
        for stat in stat_list:
            agg_dict[name+stat] = stat
        result = train[base_feat + other_feat].groupby(base_feat)[",".join(other_feat)]\
        .agg(agg_dict)
        result = left_merge(train,result,on=base_feat)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result
#app_channel_device_ip_os  vs click_time 

#to cat
def get_feat_ratio_feat(base_feat,other_feat):
    """A在B中出现的比例"""
    result_path = cache_path + ('_').join(base_feat)+'_ratio_'+('_').join(other_feat)+'.hdf'
    name = ('_').join(base_feat)+'_ratio_'+('_').join(other_feat) + '_ratio_'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        base = get_feat_size(base_feat)
        other = get_feat_size(other_feat)
        result = base/other.values
        result.columns = [('_').join(base_feat)+'_ratio_'+('_').join(other_feat)]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

###################################################
#ngroup
#to cat
def get_feat_ngroup(base_feat):
    name = ('_').join(base_feat)+'_ngroup'
    result_path = cache_path + ('_').join(base_feat)+'_ngroup'+'.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        train[name] = train.groupby(base_feat).ngroup()
        result = train[[name]]
        train.drop([name],axis=1,inplace=True)        
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


def get_click_next_hash(base_feat):
    name = ('_').join(base_feat)+'_hash_click'
    result_path = cache_path + ('_').join(base_feat)+'_hash_click'+'.hdf'
    print(name)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        D=2**26
        train['category'] = '-'
        for i in base_feat:
            train['category'] += train[i].astype(str)+ '_'
        train['category'] = train['category'].apply(lambda x:hash(x[1:-1])% D)
        click_buffer= np.full(D, 3000000000, dtype=np.uint32)
        train['epochtime']= train['click_time'].astype(np.int64) // 10 ** 9
        next_clicks= []
        for category, t in zip(reversed(train['category'].values), reversed(train['epochtime'].values)):
            next_clicks.append(click_buffer[category]-t)
            click_buffer[category]= t
        del(click_buffer)
        QQ= list(reversed(next_clicks))
        train[name] = QQ
        result = train[[name]]
        train.drop(['epochtime','category',name],axis=1,inplace=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

#一组的最大和最小时间差
def get_time_span(base_feat):
    result_path = cache_path + ('_').join(base_feat)+'_time_span'+'.hdf'
    name = ('_').join(base_feat)+'_time_span'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:       
        print("time_span" + result_path)
        result = (train.groupby(base_feat).click_time.max()-train.groupby(base_feat).click_time.min())/timedelta(seconds=1)
        result = result.to_frame().reset_index()
        result.columns = base_feat+[name]
        result = left_merge(train,result,on=base_feat)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        gc.collect()
    return result


def get_time_span_min(base_feat):
    result_path = cache_path + ('_').join(base_feat)+'_time_span_min'+'.hdf'
    name = ('_').join(base_feat)+'_time_span_min'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:       
        print("time_span" + result_path)
        print("time_span" + result_path)
        result = train[base_feat+['click_time']].copy()
        result = result.groupby(base_feat).click_time.min().reset_index().rename(columns={'click_time': 'min_clicktime'})
        result = left_merge(train,result,on =base_feat)
        result = (train.click_time - result['min_clicktime'])/timedelta(seconds=1)
        train[name] = result
        result = train[[name]]
        train.drop([name],axis=1,inplace=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        gc.collect()
    return result


def get_time_span_max(base_feat):
    result_path = cache_path + ('_').join(base_feat)+'_time_span_max'+'.hdf'
    name = ('_').join(base_feat)+'_time_span_max'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:       
        print("time_span" + result_path)
        result = train[base_feat+['click_time']].copy()
        result = result.groupby(base_feat).click_time.max().reset_index().rename(columns={'click_time': 'max_clicktime'})
        result = left_merge(train,result,on =base_feat)
        result = (result['max_clicktime']- train.click_time)/timedelta(seconds=1)
        train[name] = result
        result = train[[name]]
        train.drop([name],axis=1,inplace=True)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        gc.collect()
    return result


def get_time_span_average(base_feat):
    result_path = cache_path + ('_').join(base_feat)+'_timespan_average'+'.hdf'
    name = ('_').join(base_feat)+'_timespan_average'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:       
        print("_timespan_average" + result_path)
        base = get_time_span(base_feat)
        other = get_feat_size(base_feat)
        result = base/other.values
        result.columns = [name]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

#一组的最大和最小时间差
def get_last_day_stat(base_feat):
    result_path = cache_path + ('_').join(base_feat)+'_last_day_stat'+'.hdf'
    name = [('_').join(base_feat)+'_last_day_count',('_').join(base_feat)+'_last_day_ratio']
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:       
        print("last_day_stat" + result_path)
        base_feat = base_feat+['day']
        last_day_count = train[base_feat].groupby(by=base_feat).size().to_frame()
        last_day_attributed = train[base_feat+['is_attributed']].groupby(by=base_feat)['is_attributed'].sum().to_frame().fillna(0)
        last_day_count['ratio'] = last_day_attributed['is_attributed']/last_day_count[0]
        last_day_count.columns = name
        result = last_day_count.reset_index()
        result.day = result.day+1
        result = left_merge(train,result,on=base_feat)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        gc.collect()
    return result


def get_last_day_ratio(base_feat):
    result_path = cache_path + ('_').join(base_feat)+'_last_day_ratio'+'.hdf'
    name = [('_').join(base_feat)+'_last_day_count',('_').join(base_feat)+'_last_day_ratio']
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:       
        print("last_day_stat" + result_path)
        base_feat = base_feat+['day']
        last_day_count = train[base_feat].groupby(by=base_feat).size().to_frame()
        last_day_attributed = train[base_feat+['is_attributed']].groupby(by=base_feat)['is_attributed'].sum().to_frame().fillna(0)
        last_day_count['ratio'] = last_day_attributed['is_attributed']/(last_day_count[0] + 100)#add2
        last_day_count.columns = name
        result = last_day_count.reset_index()
        result.day = result.day+1
        result = left_merge(train,result,on=base_feat)
        result = result[[name[-1]]]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        gc.collect()
    return result

def get_last_day_count(base_feat):
    result_path = cache_path + ('_').join(base_feat)+'_last_day_count'+'.hdf'
    name = [('_').join(base_feat)+'_last_day_count',('_').join(base_feat)+'_last_day_ratio']
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:       
        print("last_day_stat" + result_path)
        base_feat = base_feat+['day']
        last_day_count = train[base_feat].groupby(by=base_feat).size().to_frame()
        last_day_attributed = train[base_feat+['is_attributed']].groupby(by=base_feat)['is_attributed'].sum().to_frame().fillna(0)
        last_day_count['ratio'] = last_day_attributed['is_attributed']/last_day_count[0]
        last_day_count.columns = name
        result = last_day_count.reset_index()
        result.day = result.day+1
        result = left_merge(train,result,on=base_feat)
        result[name[0]] = result[name[0]].apply(lambda x: np.log(1 + x)) #add2
        result = result[[name[0]]]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        gc.collect()
    return result

def get_attri_minus(base_feat = ['app']):
    result_path = cache_path + ('_').join(base_feat)+'_attri_minus'+'.hdf'
    name = [('_').join(base_feat)+'get_attri_minus',('_').join(base_feat)+'get_attri_minus']
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:       
        print("last_day_stat" + result_path)
        result = train[base_feat+['click_time','attributed_time']].copy()
        result['minus'] = (result['attributed_time'] - result['click_time']+ timedelta(hours=8))/timedelta(seconds=1)
        result.drop(['attributed_time','click_time'],axis=1,inplace=True)      
        result = result[base_feat+ ['minus']].groupby(by=base_feat)['minus'].median().reset_index()
        result = left_merge(train,result,on=base_feat)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        gc.collect()
    return result


#二阶统计
def get_feat_stat_feat_2level(base_feat,other_feat,stat_list=['var','size']):
    result_path = cache_path + ('_').join(base_feat)+'_'+('_').join(stat_list)+'_'+('_').join(other_feat)+'_2level'+'.hdf'
    name = ('_').join(base_feat) +'_'+('_').join(other_feat)+'_'+'2level'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
        print("get_stat" + result_path)
        agg_dict = {}
        for stat in stat_list:
            agg_dict[name+stat] = stat
        result = train[base_feat].groupby(base_feat).size().reset_index().groupby(other_feat)[0]\
        .agg(agg_dict)
        result = left_merge(train,result,on=other_feat)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

#上一分钟内的点击次数
def get_last_time_click(base_feat,roll_time='1min'):
    result_path = cache_path + ('_').join(base_feat)+'_last_time_click'+'_'+roll_time+'.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        print(result_path)
        result = train[base_feat+['day','click_time']].copy()
        result.index=result.click_time
        result = result.groupby(base_feat)['day'].rolling(roll_time).count()\
        .reset_index().rename(columns={'day': ('_').join(base_feat)+'_last_time_click'+'_'+roll_time})
        result = result.groupby(base_feat+['click_time'])[('_').join(base_feat)+'_last_time_click'+'_'+roll_time].max().reset_index()
        result = left_merge(train,result,on=base_feat+['click_time'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        gc.collect()
    return result

def get_click_next_easy(base_feat):
    def days_hours_minutes(td):
        return td.seconds//3600
    name = "_".join(base_feat) + "_click_easy"
    result_path = name + '.hdf'
    print(result_path)
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        result =-1 * train[base_feat + ['click_time']].groupby(base_feat,group_keys=False)['click_time'].diff(-1).reset_index().rename(columns={'click_time': name})
        result[name] = result[name].apply(lambda x: days_hours_minutes(x))
        result = result[[name]]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
        gc.collect()
    return result

###################################################
#            make_feats
###################################################
all_data_frame = []
all_data_frame.append(train)

import multiprocessing as mp
from time import strftime,gmtime

pool = mp.Pool()
result = []
result_name = []


ratio_list = [
    
]
for feats in ratio_list:
    result.append(pool.apply_async(get_feat_ratio_feat, [feats[:-1]]))


count_list = [
    ['channel'], #add
    ['app'],
    ['channel','app'], #add 2
    ['device','os'],
    ['device','os','hour'],
    ['device','os','15min'],
    ['ip'],
    ['ip','day','hour'], 
    ['ip','day','15min'], 
    ['ip','day','3min'], 
    ['ip', 'app'],
    ['ip','app', 'os'],
    ['ip','app','day','hour'],  #add
    #['ip','device','app','day','hour'], #add
    #['ip','device','os','app','day','hour'], #add
    ['ip','device','os','app','day','15min'], #add
    ['ip','device','os','app','channel'], #add
]
#count 要改一下。。
for feats in count_list:
    result.append(pool.apply_async(get_feat_size, [feats]))

feat_size_feat = [
    ['ip', 'channel'],
    #['app', 'channel'], 
    ['channel', 'app'], 
    #['channel', 'ip'], # channle中出现了几个ip
    ['ip', 'day', 'hour'], #筛掉
    ['ip', 'day', '15min'], #筛掉
    ['ip', 'day', '3min'], #筛掉
    ['ip', 'day'],
    ['day', 'ip'], #ip出现过几天
    ['ip', 'app'],
    ['ip', 'app', 'os'], 
    ['ip', 'device'],
    #['app', 'channel'],#特征筛选去掉了
    ['ip', 'device', 'os', 'app'],
    ['ip', 'device', 'os', 'channel'], #add 2
]

for feats in feat_size_feat:
    result.append(pool.apply_async(get_feat_size_feat, [feats, feats[:-1]]))

result.append(pool.apply_async(get_feat_size_feat, [['ip', 'day', 'hour'], ['ip']]))
result.append(pool.apply_async(get_feat_size_feat, [['ip', 'day', '15min'], ['ip']]))
result.append(pool.apply_async(get_feat_size_feat, [['ip', 'day', '3min'], ['ip']]))
#result.append(pool.apply_async(get_feat_size_feat, [['channel', 'day', 'hour'], ['channel']])) #add 2
result.append(pool.apply_async(get_feat_size_feat, [['ip', 'channel', 'os'], ['ip']]))

pool.close()
pool.join()

for aresult in result:
    if SMALL:
        all_data_frame.append(aresult.get())
    else:
        pass

print("done count AND unique feature")

gc.collect()


##每个A分组下，B的均值和方差
pool = mp.Pool()
stat_list = [
    #['ip', 'day', 'device','hour','var'], #筛掉了
    #['ip', 'day', 'channel','hour','var'], #筛掉了
    ['ip', 'day', 'channel','hour','mean'],
    ['ip', 'app', 'os', 'hour','var'],
    #['ip', 'app', 'channel', 'day','var'], #筛掉了
    #['ip', 'device','app', 'os', 'hour','var'],
    #['ip', 'device', 'day', 'hour','min'], ##day,hour
    #['ip', 'device', 'day', 'hour','max'], ##day,hour
    #['ip', 'device', 'day', 'hour','mean'], ##day,hour
    ['ip', 'device', 'day', '15min','mean'], ##day,hour
    ['ip', 'device', 'day', '3min','mean'], ##day,hour
    #['ip', 'device', 'channel', 'day', 'hour','mean'], ##day,hour add2
    #['device', 'channel', 'day', 'hour','mean'], ##day,hour add2
    #['app', 'day', 'hour','mean'], ##day,hour
    #['ip', 'device','os', 'hour','min'], ##day,hour
    #['ip', 'device','os', 'hour','max'], ##day,hour
    #['ip', 'device', 'hour','min'], ##day,hour
    #['ip', 'hour','min'], ##day,hour
]
result = []
for feats in stat_list:
    result.append(pool.apply_async(get_feat_stat_feat, [feats[:-2], feats[-2:-1], feats[-1:] ]))
#result.append(pool.apply_async(get_feat_stat_feat, [['ip', 'day', 'de', feats[-2:-1], feats[-1:] ]))   
pool.close()
pool.join()
for aresult in result:
    if SMALL:
        all_data_frame.append(aresult.get())
    else:
        pass

gc.collect()

## 每个A组下存在B数量的均值，方差
pool = mp.Pool()
stat_list = [
    #['ip', 'app','mean'],
    #['ip', 'channel','mean'],
    ['ip', 'device','app','mean'],
    ['channel', 'device','app','mean'],
    ['device','os','app','channel','mean'],
    ['os','ip','mean'],
  #  ['channel', 'os','app','mean'], #add 2
]
result = []
for feats in stat_list:
    result.append(pool.apply_async(get_feat_stat_feat_2level, [feats[:-1], feats[-2:-1], feats[-1:] ]))
#result.append(pool.apply_async(get_feat_stat_feat_2level, [['ip', 'app','channel'], ['app','channel'],['max']])) #add 100
result.append(pool.apply_async(get_feat_stat_feat_2level, [['ip', 'app','channel'], ['app','channel'],['var']])) #add 100
result.append(pool.apply_async(get_feat_stat_feat_2level, [['ip','device', 'app','channel'], ['app','channel'],['mean']]))
#result.append(pool.apply_async(get_feat_stat_feat_2level, [['ip','device', 'os','app'],['app'],['mean']])) #筛掉了
#result.append(pool.apply_async(get_feat_stat_feat_2level, [['ip','device', 'os','app','channel'],['app','channel'],['mean']]))

pool.close()
pool.join()
for aresult in result:
    if SMALL:
        all_data_frame.append(aresult.get())
    else:
        pass

gc.collect()


rank_list_true = [
    #['ip'],
    #['ip','hour'], #done
    #['ip', 'device','app'], #筛掉
    #['ip', 'device', 'app', 'channel'],
    #['ip', 'app'], #筛掉
    #['ip', 'app', 'channel'],# add
    ['ip', 'app', 'device', 'os']# add
]
result = []
pool = mp.Pool()
for feats in rank_list_true:
    result.append(pool.apply_async(get_rank, [feats]))

rank_list_false = [
    ['ip'],
    ['ip','hour'],
    #['ip','30min'],
    #['ip', 'device','app'], #筛掉
    #['ip', 'device', 'app', 'channel'], #筛掉
    #['ip', 'app'], #筛掉
    #['ip', 'app', 'channel'],# add
    #['ip', 'app', 'device', 'os']# add
]
for feats in rank_list_false:
    result.append(pool.apply_async(get_rank, [feats, False]))

pool.close()
pool.join()
for aresult in result:
    if SMALL:
        all_data_frame.append(aresult.get())
    else:
        pass

print("done2")
gc.collect()



click_list_next = [
    #['ip','app'],
    ['ip','app','device'],
    #['ip', 'app', 'channel'], # add
    #['ip', 'app', 'os'], #add
    ['ip','device','os','app'], # add user
]

result = []
pool = mp.Pool()
for feats in click_list_next:
    result.append(pool.apply_async(get_next_click_time, [feats]))
    #result.append(pool.apply_async(get_last_time_click, [feats]))
    #result.append(pool.apply_async(get_click_next_easy, [feats]))
result.append(pool.apply_async(get_click_next_hash, [['ip','device','os','app']])) #新的特征 


click_list_last = [
    #['ip','app'],
    #['ip', 'app', 'channel'], # add
    ['ip', 'app', 'os'], #add
    ['ip','device','os','app'], # add user
]
for feats in click_list_last:
    result.append(pool.apply_async(get_next_click_time, [feats,'last']))

pool.close()
pool.join()
for aresult in result:
    if SMALL:
        tmp_tr = aresult.get()
        all_data_frame.append(tmp_tr.fillna(99999))
    else:
        pass
gc.collect() 

n_group_list = [
    ['app','channel']
]
#span
span_list = [
    ['ip','day'],
    #['app'], #筛掉了
    ['ip','device'],
    #['ip','os'],
    #['ip','app','device'],
    ['ip','os','device'],
    #['ip', 'app', 'channel'], # add
    #['ip', 'app', 'os'], #add
]

result = []
pool = mp.Pool()
for feats in span_list:
    result.append(pool.apply_async(get_time_span, [feats]))

# for feats in n_group_list:
#     result.append(pool.apply_async(get_feat_ngroup, [feats]))

pool.close()
pool.join()
for aresult in result:
    if SMALL:
        all_data_frame.append(aresult.get())
    else:
        pass
gc.collect() 


#span
span_list_averge = [
    ['ip'],
    #['app'],
    ['ip','device'],
    #['ip','os'],
    #['ip','app','device'],
    ['ip','os','device'],
    #['app','hour'],
    #['app','day'],
    ['os','device'],
    #['ip', 'device', 'channel','day'], # add 2
    #['ip', 'app', 'os'], #add
]
result = []
pool = mp.Pool()
for feats in span_list_averge:
    result.append(pool.apply_async(get_time_span_average, [feats]))

pool.close()
pool.join()
for aresult in result:
    if SMALL:
        all_data_frame.append(aresult.get())
    else:
        pass
gc.collect() 



#span
last_day_stat_list = [
    ['ip'],
    ['ip','device'],
    ['channel'],
    #['ip','os'],
    #['app'],
    #['ip','device'], #add
    #['ip','device','os'], #add
    #['ip','app','device'],
]

result = []
pool = mp.Pool()
for feats in last_day_stat_list:
    result.append(pool.apply_async(get_last_day_ratio, [feats])) #get_last_day_ratio
    #result.append(pool.apply_async(get_last_day_count, [feats])) #get_last_day_count
pool.close()
pool.join()
for aresult in result:
    if SMALL:
        all_data_frame.append(aresult.get())
    else:
        pass
gc.collect() 

if (SMALL == False):
    exit()
train = concat(all_data_frame)
del all_data_frame
gc.collect()

target = 'is_attributed'
#############other feature########################

#train['ip_count'] = train['ip_count'].apply(lambda x: np.log(1+x))

train['ip_count_app_median'] = get_feat_stat_feat(['app'],['ip_count'],['median'])
train['ratio1'] = get_feat_ratio_feat(["app"],["app","channel"])
# df = pd.get_dummies(train['hour'])
# for f in df.columns:
#     train[str(f)] = df[f]
#train['ip_device_os_app_next_click_next_var'] = get_feat_stat_feat("ip_device_os_app".split('_'), ['ip_device_os_app_next_click_next'], ['var'])
#train['ip_device_os_app_next_click_next_min'] = get_feat_stat_feat("ip_device_os_app".split('_'), ['ip_device_os_app_next_click_next'], ['min'])

predictors = [f for f in train.columns if f not in ['click_time','is_attributed','click_id','ip','day','attributed_time','minute','3min','15min','30min']]

categorical = ['app', 'channel', 'device', 'os', 'hour'] + [f for f in train.columns if f[-6:] == 'ngroup']

# train['ip_day_count'] = train['ip_day_count'].astype('uint16')
# train['ip_count'] = train['ip_count'].astype('uint16')
# train['ip_app_count'] = train['ip_app_count'].astype('uint16')     

def lgb_modelfit_nocv( dtrain, dvalid, predictors, target='target', objective='binary', metrics=['auc', 'logloss'],
                 feval=None, early_stopping_rounds=20, num_boost_round=30, verbose_eval=20, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':{'binary_logloss', 'auc'},
        'learning_rate': 0.1,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 2**6 - 2,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 6,  # -1 means no limit
        #'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 155,  # Number of bucketed bin for feature values
        'subsample': 0.75,  # Subsample ratio of the training instance.
        #'subsample_freq': 0.9,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.75,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'verbose': 0,
        'scale_pos_weight':200,
    }

    #lgb_params.update(params)

    print("preparing validation datasets")
    print("train size")
    print(len(train.iloc[:third_day_start_index]))
    print("test size")
    print(len(train.iloc[third_day_start_index:]))                    
    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics[0]+":", evals_results['valid'][metrics[0]][bst1.best_iteration-1])
    gain = bst1.feature_importance('gain')
    ft = pd.DataFrame({'feature':bst1.feature_name(),'split':bst1.feature_importance('split'),'gain':100*gain/gain.sum()}).sort_values('gain',ascending=False)
    print(ft.head(60))
    if DEBUG:
        ft.to_csv(cache_path + 'importance_lightgbm_{}.csv'.format(evals_results['valid'][metrics[0]][bst1.best_iteration-1]),index=True)
    else:
        ft.to_csv('importance_lightgbm_{}.csv'.format(evals_results['valid'][metrics[0]][bst1.best_iteration-1]),index=True)
    return (bst1,bst1.best_iteration,evals_results['valid'][metrics[0]][bst1.best_iteration-1])

# params = {
#     'learning_rate': 0.20,
#     #'is_unbalance': 'true', # replaced with scale_pos_weight argument
#     'num_leaves': 7,  # 2^max_depth - 1
#     'max_depth': 3,  # -1 means no limit
#     'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
#     'max_bin': 100,  # Number of bucketed bin for feature values
#     'subsample': 0.7,  # Subsample ratio of the training instance.
#     'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
#     'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
#     'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
#     'scale_pos_weight':200 # because training data is extremely unbalanced 
# }
(bst,best_iteration,score) = lgb_modelfit_nocv( 
                        train.iloc[:third_day_start_index], 
                        train.iloc[third_day_start_index:], 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics=['auc','logloss'],
                        early_stopping_rounds=30, 
                        verbose_eval=True, 
                        num_boost_round=10000, 
                        categorical_features=categorical)

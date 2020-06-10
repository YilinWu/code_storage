import pandas as pd
import time
import numpy as np
import psutil
import os
import gc
import lightgbm as lgb
import pytz


path = ''
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
train = pd.read_csv(path+"train.csv", dtype=dtypes,usecols=train_cols, parse_dates=['click_time'])#,nrows=100000)
test = pd.read_csv(path+"test_supplement.csv", parse_dates=['click_time'],dtype=dtypes, usecols=test_cols)#,nrows=100000)

def click_time(df):
    #cst = pytz.timezone('Asia/Shanghai')
    #df['click_time'] = pd.to_datetime(df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
    df['day'] = df.click_time.dt.day.astype('uint8')
    #df['hour'] = df.click_time.dt.hour.astype('uint8')
    return df

train_rc = click_time(train)
test_rc = click_time(test)

def left_merge(data1,data2,on):
    data1['click_time']=pd.datetime(2050,1,1) - data1['click_time']

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

def get_last_time_click(train,base_feat,roll_time ='1min'):
	result = train[base_feat+['day','click_time']].iloc[::-1].copy()
	# result1 = train_rc[base_feat+['day','click_time']].iloc[::-1].copy()
	result.index=result['click_time']

	result.index = pd.datetime(2050,1,1) - result.click_time

	result = result.groupby(base_feat)['day'].rolling(roll_time).count()[::-1]\
	.reset_index().rename(columns = {'day':('_').join(base_feat)+'_future_click'+'_'+roll_time})

	#result.index = pd.datetime(2050,1,1) - result.click_time

	result=result.groupby(base_feat+['click_time'])[('_').join(base_feat)+'_future_click'+'_'+roll_time].max().reset_index()
	result = left_merge(train, result,on=base_feat+['click_time'])
	#final = result.drop('click_time',axis =1).reset_index()

	gc.collect()
	return result

print('start 2min')
train_recent_2min = get_last_time_click(train_rc,['ip','device','os'],roll_time = '2min')
test_recent_2min = get_last_time_click(test_rc,['ip','device','os'],roll_time = '2min')
data_recent_2min = pd.concat([train_recent_2min,test_recent_2min])
del train_recent_2min, test_recent_2min
data_recent_2min.to_pickle('data/feature_future_3g_2min_data.pkl')
print(data_recent_2min.shape)
del data_recent_2min
gc.collect()
print('finish 2min')


print('start 15min')
train_recent_15min = get_last_time_click(train_rc,['ip','device','os'],roll_time = '15min')
test_recent_15min = get_last_time_click(test_rc,['ip','device','os'],roll_time = '15min')
data_recent_15min = pd.concat([train_recent_15min,test_recent_15min])
del train_recent_15min, test_recent_15min
data_recent_15min.to_pickle('data/feature_future_3g_15min_data.pkl')
print(data_recent_15min.shape)
del data_recent_15min
gc.collect()
print('finish 30min')


print('start 30min')
train_recent_30min = get_last_time_click(train_rc,['ip','device','os'],roll_time = '30min')
test_recent_30min = get_last_time_click(test_rc,['ip','device','os'],roll_time = '30min')
data_recent_30min = pd.concat([train_recent_30min,test_recent_30min])
del train_recent_30min, test_recent_30min
data_recent_30min.to_pickle('data/feature_future_3g_30min_data.pkl')
print(data_recent_30min.shape)
del data_recent_30min
gc.collect()
print('finish 30min')

print('start 60min')
train_recent_60min = get_last_time_click(train_rc,['ip','device','os'],roll_time = '60min')
test_recent_60min = get_last_time_click(test_rc,['ip','device','os'],roll_time = '60min')
data_recent_60min = pd.concat([train_recent_60min,test_recent_60min])
del train_recent_60min, test_recent_60min
data_recent_60min.to_pickle('data/feature_future_3g_60min_data.pkl')
print(data_recent_60min.shape)
del data_recent_60min
gc.collect()
print('finish 60min')	
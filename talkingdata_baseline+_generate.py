"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels
This kernel gives LB: 0.977 when the parameter 'debug' below is set to 0 but this implementation requires a machine with ~32 GB of memory
"""
import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score
path='C:/Users/misaka/Downloads/Adtalkingdata/'
#warnings.filterwarnings("ignore")

import pytz
import gc

import random
import scipy.special as special
from tqdm import tqdm

np.random.seed(0)

class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in tqdm(range(iter_num)):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            #print (new_alpha,new_beta,i)
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)


import time

# data frame
def merge_count_for_lag(df,df2,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    add.columns=columns+[cname]
    df2=df2.merge(add,on=columns,how="left")
    return df2

def merge_mean_for_lag(df,df2,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns=columns+[cname]
    df2=df2.merge(add,on=columns,how="left")
    return df2   

def merge_sum_for_lag(df,df2,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    add.columns=columns+[cname]
    df2=df2.merge(add,on=columns,how="left")
    return df2

def merge_count(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df


def merge_nunique(df,columns,value,cname): # number of unique()
    add = pd.DataFrame(df.groupby(columns)[value].nunique()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_median(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_mean(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_sum(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_max(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].max()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_min(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].min()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_std(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].std()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def interaction_features(train, fea1, fea2, e):
    train['inter_{}_{}_{}*'.format(e,fea1,fea2)] = train[fea1] * train[fea2]
    train['inter_{}_{}_{}/'.format(e,fea1,fea2)] = train[fea1] / train[fea2]
    train['inter_{}_{}_{}+'.format(e,fea1,fea2)] = train[fea1] + train[fea2]
    train['inter_{}_{}_{}-'.format(e,fea1,fea2)] = train[fea1] - train[fea2]
    return train

def interaction_features2(train, fea1, fea2, e1,e2):
    train['inter_{}_{}*'.format(e1,e2)] = train[fea1] * train[fea2]
    train['inter_{}_{}/'.format(e1,e2)] = train[fea1] / train[fea2]
    train['inter_{}_{}+'.format(e1,e2)] = train[fea1] + train[fea2]
    train['inter_{}_{}-'.format(e1,e2)] = train[fea1] - train[fea2]
    return train

#这里还可以加 其他组合，时间： C82=28


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def map_hour(x):
    if (x>=7)&(x<=11): #7-13点
        return 1
    elif (x>11)&(x<=17):
        return 2
    elif (x>=18)&(x<=20): #13-21点
        return 3
    elif (x>=21)&(x<=24): #13-21点
        return 4
    else:
        return 5

#特征：

def base_process(data):
    
    return data


def timefeature(data):
    data['hour_map'] = data['hour'].apply(map_hour)
    return data

def statistics(data): 

    print('stats processing...')
  
    shopcnt = data.groupby(['shop_id'], as_index=False)['instance_id'].agg({'shop_cnt': 'count'}) # 我日？？
    data = pd.merge(data, shopcnt, on=['shop_id'], how='left')

    data = merge_count(data,['shop_review_num_level'],"instance_id","shop_review_num_level_cnt")
    data = merge_count(data,['shop_star_level'],"instance_id","shop_star_level_cnt")

    data=merge_mean(data,['user_id','item_id'],"shop_review_positive_rate","shop_review_positive_mean")
    data=merge_mean(data,['user_id','item_id'],"shop_score_service","shop_score_service_mean")
    data=merge_mean(data,['context_page_id','item_id'],"item_sales_level","item_sales_mean")
    data=merge_mean(data,['context_page_id','item_id'],"shop_star_level","shop_star_mean")

    data=merge_median(data,['shop_id'],"shop_review_positive_rate","shop_review_median")
    data=merge_median(data,['shop_id'],"shop_score_service","shop_service_median")
    data=merge_median(data,['shop_id'],"shop_score_delivery","shop_delivery_median")
    data=merge_median(data,['shop_id','item_id'],"shop_score_description","shop_description_median")

    data=merge_mean(data,['shop_id'],"shop_review_positive_rate","shop_review_mean")
    data=merge_mean(data,['shop_id'],"shop_score_service","shop_service_mean")
    data=merge_mean(data,['shop_id'],"shop_score_delivery","shop_delivery_mean")
    data=merge_mean(data,['shop_id'],"shop_score_description","shop_description_mean")

    data=merge_median(data,['item_id'],"shop_review_positive_rate","item_review_median")
    data=merge_median(data,['item_id'],"shop_score_service","item_service_median")
    data=merge_median(data,['item_id'],"shop_score_delivery","item_delivery_median")
    data=merge_median(data,['item_id'],"shop_score_description","item_description_median")

    data=merge_mean(data,['item_id'],"shop_review_positive_rate","item_review_mean")
    data=merge_mean(data,['item_id'],"shop_score_service","item_service_mean")
    data=merge_mean(data,['item_id'],"shop_score_delivery","item_delivery_mean")
    data=merge_mean(data,['item_id'],"shop_score_description","item_description_mean")

    data=merge_mean(data,['user_id'],"shop_review_positive_rate","user_review_mean")
    data=merge_mean(data,['user_id'],"shop_score_service","user_service_mean")
    data=merge_mean(data,['user_id'],"shop_score_delivery","user_delivery_mean")
    data=merge_mean(data,['user_id'],"shop_score_description","user_description_mean")

    for e, (x, y) in enumerate(combinations(['shop_review_positive_rate', 'shop_score_service', 
                      'shop_score_delivery', 'shop_score_description'], 2)): 
        data = interaction_features(data, x, y, e) #1


    return data

def history_cnt(data): 

    print('当前日期前一天的cnt/cvr')  # 18-23 validate 24 test 25
    

    print('当前日期之前的cnt/cvr') # 789 10 # 10号is_attributed=nan
    for d in range(8, 11):
        df1 = data[data['day'] < d] #18
        df2 = data[data['day'] == d] 

        df2=merge_count_for_lag(df1,df2,['ip'],'is_attributed','ip_cntx') #点击次数
        df2=merge_mean_for_lag (df1,df2,['ip'],'is_attributed','ip_cvrx')  #转化率
        df2=merge_sum_for_lag  (df1,df2,['ip'],'is_attributed','ip_cvtimesx') #转化次数

        df2=merge_count_for_lag(df1,df2,['device'],'is_attributed','device_cntx') #点击次数
        df2=merge_mean_for_lag (df1,df2,['device'],'is_attributed','device_cvrx')  #转化率
        df2=merge_sum_for_lag  (df1,df2,['device'],'is_attributed','device_cvtimesx') #转化次数

        df2=merge_count_for_lag(df1,df2,['os'],'is_attributed','os_cntx') #点击次数
        df2=merge_mean_for_lag (df1,df2,['os'],'is_attributed','os_cvrx')  #转化率
        df2=merge_sum_for_lag  (df1,df2,['os'],'is_attributed','os_cvtimesx') #转化次数

        df2=merge_count_for_lag(df1,df2,['app'],'is_attributed','app_cntx') #点击次数
        df2=merge_mean_for_lag (df1,df2,['app'],'is_attributed','app_cvrx')  #转化率
        df2=merge_sum_for_lag  (df1,df2,['app'],'is_attributed','app_cvtimesx') #转化次数

        df2=merge_count_for_lag(df1,df2,['channel'],'is_attributed','channel_cntx') #点击次数
        df2=merge_mean_for_lag (df1,df2,['channel'],'is_attributed','channel_cvrx')  #转化率
        df2=merge_sum_for_lag  (df1,df2,['channel'],'is_attributed','channel_cvtimesx') #转化次数

        df2=merge_count_for_lag(df1,df2,['ip','app'],'is_attributed','ip_app_cntx') #点击次数
        df2=merge_mean_for_lag (df1,df2,['ip','app'],'is_attributed','ip_app_cvrx')  #转化率
        df2=merge_sum_for_lag  (df1,df2,['ip','app'],'is_attributed','ip_app_cvtimesx') #转化次数

        df2=merge_count_for_lag(df1,df2,['channel','app'],'is_attributed','channel_app_cntx') #点击次数
        df2=merge_mean_for_lag (df1,df2,['channel','app'],'is_attributed','channel_app_cvrx')  #转化率
        df2=merge_sum_for_lag  (df1,df2,['channel','app'],'is_attributed','channel_app_cvtimesx') #转化次数

        df2=merge_count_for_lag(df1,df2,['device','app'],'is_attributed','device_app_cntx') #点击次数
        df2=merge_mean_for_lag (df1,df2,['device','app'],'is_attributed','device_app_cvrx')  #转化率
        df2=merge_sum_for_lag  (df1,df2,['device','app'],'is_attributed','device_app_cvtimesx') #转化次数

        df2=merge_count_for_lag(df1,df2,['ip','channel'],'is_attributed','ip_channel_cntx') #点击次数
        df2=merge_mean_for_lag (df1,df2,['ip','channel'],'is_attributed','ip_channel_cvrx')  #转化率
        df2=merge_sum_for_lag  (df1,df2,['ip','channel'],'is_attributed','ip_channel_cvtimesx') #转化次数
      

        df2=df2[['ip_cntx','device_cntx','os_cntx','app_cntx','channel_cntx','ip_app_cntx','channel_app_cntx',\
        'ip_channel_cntx','ip_cvrx','device_cvrx','os_cvrx','app_cvrx','channel_cvrx','ip_app_cvrx','channel_app_cvrx',\
        'ip_channel_cvrx','ip_cvtimesx','device_cvtimesx','os_cvtimesx','app_cvtimesx',\
        'channel_cvtimesx','ip_app_cvtimesx','channel_app_cvtimesx','ip_channel_cvtimesx',\
        'ip','os','device','app','channel','click_time']]
        if d == 8:     
            Df2 = df2
        else:            
            Df2 = pd.concat([df2, Df2]) #
    data = pd.merge(data, Df2, on=['ip','os','device','app','channel','click_time'], how='left')

    print("history_cvr_without_smooth")

    return data
      

def time_delta(data): # 按天划分一下：：：： 按dow 划分下
    
    data['click_time'] = pd.to_datetime(data['click_time'])

    data['ip_delta'] = data.groupby('ip').click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    data['app_delta'] = data.groupby('app').click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds # 强力输出
    data['channel_delta'] = data.groupby('channel').click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    data['ip_app_delta'] = data.groupby(['ip','app']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    data['ip_channel_delta'] = data.groupby(['ip','channel']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    data['app_channel_delta'] = data.groupby(['app','channel']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    data['ip_app_channel_delta'] = data.groupby(['ip','app','channel']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    data['ip_os_app_delta'] = data.groupby(['ip','os','app'])['click_time'].diff().shift(-1).dt.seconds
 #   data['5_delta'] = data.groupby(['ip','channel','app','device','os'])['click_time'].diff().shift(-1).dt.seconds
    
    return data

def running_count(data):
    HISTORY_CLICKS = {
     'a':['ip'],            #user
     'b':['device'],        #user_occupation
     'c':['os'],            #user_age
     'd':['app'],           #item
     'e':['channel'],       #shop
     'f':['ip', 'app'],
     'g':['channel', 'app'],
     'h':['ip','channel'],
     'i':['ip','channel','app'],
     'j':['device','channel','app'],
     'k':['ip','app','day']
    }
    # Go through different group-by combinations
    for fname, fset in HISTORY_CLICKS.items():     
        # Clicks in the past 
        # Already sorted
        data['prev_'+fname] = data. \
            groupby(fset). \
            cumcount(). \
            rename('prev_'+fname)

        data['future_'+fname] = data. \
            iloc[::-1]. \
            groupby(fset). \
            cumcount(). \
            rename('prev_'+fname)

    return data




def lgb_validate(train, validate, test_inf,ss):
    col = predictors
   
    X = train_df[col]
    y = train_df[target].values
    X_val = val_df[col]
    y_val = val_df[target].values
    print('Training LGBM model...')
      
    params = {
        'learning_rate': 0.1,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 15,  # 2^max_depth - 1
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 256,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 10e-3,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':402.7 # because training data is extremely unbalanced 
    }   
    lgb_params2 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
    #    'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.6,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 10,  # L2 regularization term on weights
        'nthread': 50,
        'verbose': 1,
    }

    lgb_params2.update(params)

    dtrain = lgb.Dataset(X, y,categorical_feature=categorical) #'auto'
    dvalid = lgb.Dataset(X_val, y_val, reference=dtrain,categorical_feature=categorical)
    bst = lgb.train(lgb_params2, dtrain, num_boost_round=1000, valid_sets=[dtrain,dvalid], verbose_eval=10, # dvalid
                            early_stopping_rounds=30) #200 #125很不错配合0.03
    pred_oof = bst.predict(X_val, num_iteration=bst.best_iteration)

    # print("Features importance for training...")
    # gain = bst.feature_importance('gain')
    # ft = pd.DataFrame({'feature':bst.feature_name(), 'split':bst.feature_importance('split'), 
    #     'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
   # print(ft.tail(35))

    val_df['pred'] = pred_oof
    val_df['index'] = range(len(val_df))
    pd.DataFrame({'index':val_df['index'], 'day_24_oof':val_df['pred'] }).to_csv(path+'oof/416_oof_2_td.csv', index=False)   
    #print('误差 ', roc_auc_score(val_df['is_attributed'], val_df['pred']))

    return bst.best_iteration,pred_oof

def final_result(maxround,ss):
    col = predictors
    print('Test LGBM result...')
    params = {
        'learning_rate': 0.1,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 15,  # 2^max_depth - 1
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 256,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 10e-3,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':402.7 # because training data is extremely unbalanced 
    }
    
    lgb_params2 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
     #   'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.6,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 10,  # L2 regularization term on weights
        'nthread': 50,
        'verbose': 1,
    }

    lgb_params2.update(params)

    #+dropout in tesing
    Maxrou=round(maxround * 1.46) #  previous 1.05
    dtrain = lgb.Dataset(train_all[col], train_all[target].values,categorical_feature=categorical) #'auto'
 #   dvalid = lgb.Dataset(X_val, y_val, reference=dtrain,categorical_feature='auto')
    bst = lgb.train(lgb_params2, dtrain, num_boost_round=Maxrou, valid_sets=dtrain, verbose_eval=20)
    pred_sub = bst.predict(test_df2[col]) # suppliment

   # print("Features importance...")
   # gain = bst.feature_importance('gain')
   #ft = pd.DataFrame({'feature':bst.feature_name(), 'split':bst.feature_importance('split'), 
   #     'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
   # print(ft.head(35))
    
    print('projecting prediction onto test')

    join_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
    all_cols = join_cols + ['is_attributed']

    test = test_df.merge(test_df2[all_cols], how='left', on=join_cols)

    test = test.drop_duplicates(subset=['click_id'])

    print("Writing the submission data into a csv file...")

    test[['click_id', 'is_attributed']].to_csv('sub_holishit1.csv', index=False)

    print("All done...")

    return pred_sub

def DO(frm,to,fileno):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    print('loading train data...')
    train_df = pd.read_csv(path+"train.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

    print('loading test data...')
    if debug:
        test_df = pd.read_csv(path+"test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv(path+"test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    test_sup = pd.read_csv(path+"test_supplement.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    len_train = len(train_df)
    train_df=train_df.append(test_sup)
    
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    

    
    naddfeat=9
    for i in range(0,naddfeat):
        if i==0: selcols=['ip', 'channel']; QQ=4;
        if i==1: selcols=['ip', 'device', 'os', 'app']; QQ=5;
        if i==2: selcols=['ip', 'day', 'hour']; QQ=4;
        if i==3: selcols=['ip', 'app']; QQ=4;
        if i==4: selcols=['ip', 'app', 'os']; QQ=4;
        if i==5: selcols=['ip', 'device']; QQ=4;
        if i==6: selcols=['app', 'channel']; QQ=4;
        if i==7: selcols=['ip', 'os']; QQ=5;
        if i==8: selcols=['ip', 'device', 'os', 'app']; QQ=4;
        print('selcols',selcols,'QQ',QQ)
        
        filename='X%d_%d_%d.csv'%(i,frm,to)
        
        if os.path.exists(filename):
            if QQ==5: 
                gp=pd.read_csv(filename,header=None)
                train_df['X'+str(i)]=gp
            else: 
                gp=pd.read_csv(filename)
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        else:
            if QQ==0:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].count().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==1:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].mean().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==2:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].var().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==3:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].skew().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==4:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].nunique().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==5:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].cumcount()
                train_df['X'+str(i)]=gp.values
            
            if not debug:
                 gp.to_csv(filename,index=False)
            
        del gp
        gc.collect()    

    print('doing nextClick')
    predictors=[]
    
    new_feature = 'nextClick'
    filename='nextClick_%d_%d.csv'%(frm,to)

    if os.path.exists(filename):
        print('loading from save file')
        QQ=pd.read_csv(filename).values
    else:
        D=2**26
        train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
            + "_" + train_df['os'].astype(str)).apply(hash) % D
        click_buffer= np.full(D, 3000000000, dtype=np.uint32)

        train_df['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks= []
        for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
            next_clicks.append(click_buffer[category]-t)
            click_buffer[category]= t
        del(click_buffer)
        QQ= list(reversed(next_clicks))

        if not debug: #1
            print('saving')
            pd.DataFrame(QQ).to_csv(filename,index=False)

    train_df[new_feature] = QQ
    predictors.append(new_feature)

    train_df[new_feature+'_shift'] = pd.DataFrame(QQ).shift(+1).values  #???
    predictors.append(new_feature+'_shift')
    
    del QQ
    gc.collect()

    print('grouping by ip-day-hour combination...')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app combination...')
    gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left')
    del gp
    gc.collect()

    print('grouping by ip-app-os combination...')
    gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by : ip_day_chl_var_hour')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_os_var_hour')
    gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_channel_var_day')
    gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    del gp
    gc.collect()

    print('grouping by : ip_app_chl_mean_hour')
    gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    print("merging...")
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    del gp
    gc.collect()

    print("vars and data type: ")
    train_df.info()
    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

    target = 'is_attributed'
    predictors.extend(['app','device','os', 'channel', 'hour', 'day', 
                  'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day','ip_app_channel_mean_hour'])
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
    for i in range(0,naddfeat):
        predictors.append('X'+str(i))
        
    print('predictors',predictors)
    print('saving...')
   # train_df.to_csv(path+'another_base22_full.csv',index=False)
    print('saving complete')
    
    
    return 0
    

if __name__ == "__main__":

    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }

    print('start loading')
    train_df= pd.read_csv(path+'another_base22_full.csv')
    train_df=train_df.drop(['day','hour'],axis=1)
    test_df = pd.read_csv(path+"test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    cst = pytz.timezone('Asia/Shanghai')
    train_df['click_time'] = pd.to_datetime(train_df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
    train_df['day'] = train_df.click_time.dt.day.astype('uint8')
    train_df['hour'] = train_df.click_time.dt.hour.astype('uint8')

    print(len(train_df))
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
    predictors=['app', 'channel', 'device', 'ip', 'os', 'hour', 'day',
     'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'nextClick', 
     'nextClick_shift', 'ip_tcount', 'ip_app_count', 'ip_app_os_count', 
     'ip_tchan_count', 'ip_app_os_var', 'ip_app_channel_var_day',
     'ip_app_channel_mean_hour']
    target ='is_attributed'
    # sub=DO(frm,to,0) 
    
    #train_df=base_process(train_df)
    #train_df=timefeature(train_df)
    #train_df=statistics(train_df)
    #train_df=history_cnt(train_df)

    train_df=time_delta(train_df)
    predictors.extend(['ip_delta','app_delta','channel_delta','ip_app_delta','ip_channel_delta','app_channel_delta',\
                      'ip_app_channel_delta','ip_os_app_delta'])
    train_df=running_count(train_df)
    predictors.extend(['prev_a','prev_b','prev_c','prev_d','prev_e','prev_f','prev_g','prev_h','prev_i','prev_j','prev_k',\
                  'future_a','future_b','future_c','future_d','future_e','future_f','future_g','future_h','future_i','future_j',\
                  'future_k'])
    train_df.to_csv(path+'another_base333_full.csv',index=False)
    #train_df=interactive(train_df)

    len_train = 184903890
    print(len_train)

    print('start loading')
    print(len(train_df))
 #   train_df['minute'] = pd.to_datetime(train_df.click_time).dt.minute.astype('uint8')
    
    print(train_df.columns.tolist())

   
    train_all = train_df[train_df.is_attributed.notnull()]   #7,8,9                     
    test_df2 = train_df[(train_df['day']==10)]   #10                                     
    val_df = train_df[(train_df['day']==9)]      #9                                 
    train_df = train_df[(train_df['day']>=7)&(train_df['day']<=8)]   # 7 8                 


    print('train all',len(train_all)) # 7 8 9 预测 test_sup 10
    print('test_sup:',len(test_df2))  # 10
    print("valid size day 9: ", len(val_df)) # 9
    print("train size: ", len(train_df)) # 78   
    print("test size : ", len(test_df))

    oof_score = []
    res_score = []
    cv_pred = np.zeros(len(test_df))
    for s in range (42,52):
        bst,oof = lgb_validate(train_df,val_df,test_df,s)
        oof_score.append(oof)
        print('finish_seed_{}'.format(s))
        res = final_result(bst,s)
        cv_pred+=res
        res_score.append(res)
        print('finish_inference_{}'.format(s))
    cv_pred/=10
    

    test_df['is_attributed'] = cv_pred
    sub3 = test_df[['click_id', 'is_attributed']] 
    sub3[['click_id', 'is_attributed']].to_csv('result/average_result0416_full_02.csv',index=False)


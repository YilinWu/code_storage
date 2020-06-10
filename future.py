def cal_feature_click_cnt_field_v2(train_df, recent_time_range, field):
    train_df = train_df[[INSTANCE_ID, USER_ID, field, CONTEXT_TIMESTAMP]]
   

#    logging.info("cal_recent_click sort ")
    train_v = train_df.values.tolist()
    # train_v = sorted(train_v, key=lambda d: d[1:4])
   
    # test_v = sorted(test_v, key=lambda d: d[1:4])

    tot_v = train_v 
    tot_v = sorted(tot_v, key=lambda d: d[1:4])

    feat = []
    default_v = -1
    has_feat = 0

  #  logging.info("processing feature %s feat..." % field)
    for i, rec in enumerate(tot_v):
        if i == len(tot_v) - 1:
            one = rec[:3] + [default_v, ]

        elif rec[1:3] == tot_v[i + 1][1:3]:
            cnt = 0
            j = i + 1
            while j < len(tot_v) and tot_v[j][1:3] == rec[1:3] and tot_v[j][3] - rec[3] <= recent_time_range:
                cnt += 1
                j += 1
            one = rec[:3] + [cnt]
            if cnt > 0:
                has_feat += 1
        else:
            one = rec[:3] + [0, ]

        feat.append(one)

    key = 'real_feature_{0}_timerange_{1}'.format(field, recent_time_range)

    print ("cal:", key, " has_feat:", has_feat, "tot_rec:", len(tot_v))

 #   logging.info("cal_feature_click change2df")

    feat_df = pd.DataFrame(feat, columns=[INSTANCE_ID, USER_ID, field, key])

    # feat_df.to_csv(FEAT_DIR + key + '.csv', index=False)

    return feat_df[[INSTANCE_ID, key]]

#上面是函数下面是用法
eature_one_day_click_cnt_featxx2 = cal_feature_click_cnt_field_v2(data, ONE_DAY / (24 * 6 * 5),USER_ID) #2
data = pd.merge(data, feature_one_day_click_cnt_featxx2, how='left', on=INSTANCE_ID)
feature_one_day_click_cnt_featxx15 = cal_feature_click_cnt_field_v2(data, ONE_DAY / (24 * 4),USER_ID) #15
data = pd.merge(data, feature_one_day_click_cnt_featxx15, how='left', on=INSTANCE_ID)
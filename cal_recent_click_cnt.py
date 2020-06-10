def cal_recent_click_cnt(train_df, recent_time_range, key='recent_click_cnt'):
    train_df = train_df[[INSTANCE_ID, USER_ID, ITEM_ID, CONTEXT_TIMESTAMP]]
    train_v = train_df.values.tolist()
    tot_v = train_v 
    tot_v = sorted(tot_v, key=lambda d: d[1:4])

    feat = []
    default_v = -1
    has_feat = 0
    for i, rec in enumerate(tot_v):
        if i == 0:
            one = rec[:3] + [default_v, ]

        elif rec[1:3] == tot_v[i - 1][1:3]:
            cnt = 0
            j = i - 1
            while tot_v[j][1:3] == rec[1:3] and rec[3] - tot_v[j][3] <= recent_time_range:
                cnt += 1
                j -= 1
            one = rec[:3] + [cnt]
            if cnt > 0:
                has_feat += 1
        else:
            one = rec[:3] + [0, ]

        feat.append(one)
    key = key + '_' + str(recent_time_range) if key is not None else 'click_cnt_up_to_now'

    print ("cal:", key, " has_feat:", has_feat, "tot_rec:", len(tot_v))

    feat_df = pd.DataFrame(feat, columns=[INSTANCE_ID, USER_ID, ITEM_ID, key])

    return feat_df[[INSTANCE_ID, key]]

recent_one_day_click_cat_cnt_feat1min = cal_recent_click_cnt_field(data, ONE_DAY / (24 * 6 * 2),ITEM_CATEGORY_LIST)
data = pd.merge(data, recent_one_day_click_cat_cnt_feat1min, how='left', on=INSTANCE_ID) 
recent_one_day_click_cat_cnt_feat2min = cal_recent_click_cnt_field(data, ONE_DAY / (24 * 6 * 5),ITEM_CATEGORY_LIST)
data = pd.merge(data, recent_one_day_click_cat_cnt_feat2min, how='left', on=INSTANCE_ID)   

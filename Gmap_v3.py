#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/08/30 10:36
# @Author  : dly
# @File    : Gmap_v3.py
# @Desc    :

import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split, cross_val_score
import lightgbm as lgb
from bayes_opt import BayesianOptimization


# 读 json 数据
def get_data(df):
    map_id_list = []
    key_frame_list = []
    gap_time = []
    img_name = []
    label = []

    # 提取特征
    for s in list(df.annotations):
        map_id = s['id']
        map_key = s['key_frame']
        frames = s['frames']
        status = s['status']
        for i in range(0, len(frames)):
            f = frames[i]
            map_id_list.append(map_id)
            key_frame_list.append(map_key)
            img_name.append(f['frame_name'])
            gap_time.append(f['gps_time'])
            label.append(status)

    train_df = pd.DataFrame({'map_id': map_id_list,
                             'key_frame': key_frame_list,
                             'label': label,
                             'img_name': img_name,
                             'gap_time': gap_time})
    train_df['hour'] = train_df['gap_time'].apply(lambda x: datetime.fromtimestamp(x).hour)
    train_df['dayofweek'] = train_df['gap_time'].apply(lambda x: datetime.fromtimestamp(x).weekday())
    train_df['key_frame'] = train_df['key_frame'].apply(lambda x: int(x.split('.')[0]))

    train_df.columns = ['map_id', 'key_frame', 'label', 'img_name', 'gap_time', 'hour', 'dayofweek']
    train_df['label'] = train_df['label'].apply(int)

    print(len(train_df))

    return train_df


# 读 yolo 结果文件
def get_img_data(filepath):
    data_img = pd.read_csv(filepath, dtype=object)

    return data_img


# yolo 产生的结果文件
train_img_x = get_img_data('yolo_out/out_train.csv')
test_img_x = get_img_data('yolo_out/out_28.csv')

# json和图片数据
path = 'C:/Users/Dooooooooo21/Desktop/project/Gmap/'
train_json = pd.read_json(path + 'amap_traffic_annotations_train.json')
test_json = pd.read_json(path + 'amap_traffic_annotations_test.json')
train_df = get_data(train_json[:])
test_df = get_data(test_json[:])

# 合并两个来源的数据
train_concat_x = pd.merge(train_img_x, train_df, on=['map_id', 'img_name'])
test_concat_x = pd.merge(test_img_x, test_df, on=['map_id', 'img_name'])

# 选择数据列
select_features = ['counts', 'my_dis', 'hour', 'dayofweek']

# 训练集
X_train = train_concat_x[select_features].copy()
# 将 y 转 onehot 编码
y_train = tf.one_hot(train_concat_x["label"], 3)
X_train = pd.DataFrame(X_train, dtype=np.float)

# 测试集
X_test = test_concat_x[select_features].copy()
X_test = pd.DataFrame(X_test, dtype=np.float)


def ori_lgb():
    # 5折交叉验证
    folds = 5
    seed = 1001
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
        print('*' * 20 + str(i + 1) + '*' * 20)
        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2)
        train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
        valid_matrix = lgb.Dataset(X_val, y_val)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'multi:softmax',
            'metric': 'auc',
            'learning_rate': 0.01,
            'num_leaves': 14,
            'max_depth': 19,
            'min_data_in_leaf': 37,
            'min_child_weight': 1.6,
            'bagging_fraction': 0.98,
            'feature_fraction': 0.69,
            'bagging_freq': 96,
            'reg_lambda': 9,
            'reg_alpha': 7,
            'min_split_gain': 0.4,
            'nthread': 8,
            'seed': 2020,
        }

        model = lgb.train(params, train_set=train_matrix, valid_sets=valid_matrix, num_boost_round=14269,
                          verbose_eval=1000,
                          early_stopping_rounds=200)
        val_pre_lgb = model.predict(X_val, num_iteration=model.best_iteration)
        cv_scores.append(metrics.roc_auc_score(y_val, val_pre_lgb))

        print(cv_scores)

    print("lgb_scotrainre_list:{}".format(cv_scores))
    print("lgb_score_mean:{}".format(np.mean(cv_scores)))
    print("lgb_score_std:{}".format(np.std(cv_scores)))


# 参数优化
def rf_cv_lgb(num_leaves, max_depth, bagging_fraction, feature_fraction, bagging_freq, min_data_in_leaf,
              min_child_weight, min_split_gain, reg_lambda, reg_alpha):
    model_lgb = lgb.LGBMClassifier(boosting_type='gbdt', objective='multi:softmax', metrics='auc', learning_rate=0.1,
                                   n_estimators=5000, num_leaves=int(num_leaves), max_depth=int(max_depth),
                                   bagging_fraction=round(bagging_fraction, 2),
                                   feature_fraction=round(feature_fraction, 2),
                                   bagging_freq=int(bagging_freq), min_data_in_leaf=int(min_data_in_leaf),
                                   min_child_weight=min_child_weight, min_split_gain=min_split_gain,
                                   reg_lambda=reg_lambda, reg_alpha=reg_alpha, n_jobs=8)
    val = cross_val_score(model_lgb, X_train_split, y_train_split, cv=5, scoring='roc_auc').mean()

    return val


# 贝叶斯调参
def bayes():
    bayes_lgb = BayesianOptimization(rf_cv_lgb, {
        'num_leaves': (10, 200),
        'max_depth': (3, 20),
        'bagging_fraction': (0.5, 1.0),
        'feature_fraction': (0.5, 1.0),
        'bagging_freq': (0, 100),
        'min_data_in_leaf': (10, 100),
        'min_child_weight': (0, 10),
        'min_split_gain': (0.0, 1.0),
        'reg_alpha': (0.0, 10),
        'reg_lambda': (0.0, 10),
    })

    bayes_lgb.maximize(n_iter=10)

    # 最优参数
    print(bayes_lgb.max)


# bayes()

# 确定最优的迭代次数
def it():
    base_params_lgb = {'boosting_type': 'gbdt',
                       'objective': 'multi:softmax',
                       'metric': 'auc',
                       'learning_rate': 0.01,
                       'num_leaves': 200,
                       'max_depth': 3,
                       'min_data_in_leaf': 56,
                       'min_child_weight': 9.63,
                       'bagging_fraction': 1.0,
                       'feature_fraction': 1.0,
                       'bagging_freq': 100,
                       'reg_lambda': 10,
                       'reg_alpha': 1,
                       'min_split_gain': 1.0,
                       'nthread': 8,
                       'seed': 2020,
                       'verbose': -1, }

    cv_result_lgb = lgb.cv(train_set=train_matrix, early_stopping_rounds=1000, num_boost_round=20000, nfold=5,
                           stratified=True, shuffle=True, params=base_params_lgb, metrics='auc', seed=0)

    print('迭代次数{}'.format(len(cv_result_lgb['auc-mean'])))
    print('最终模型的AUC为{}'.format(max(cv_result_lgb['auc-mean'])))


# it()

# 最终训练模型
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2)
train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
valid_matrix = lgb.Dataset(X_val, y_val)

params = {
    'boosting_type': 'gbdt',
    'objective': 'multi:softmax',
    'metric': 'auc',
    'learning_rate': 0.01,
    'num_leaves': 14,
    'max_depth': 19,
    'min_data_in_leaf': 37,
    'min_child_weight': 1.6,
    'bagging_fraction': 0.98,
    'feature_fraction': 0.69,
    'bagging_freq': 96,
    'reg_lambda': 9,
    'reg_alpha': 7,
    'min_split_gain': 0.4,
    'nthread': 8,
    'seed': 2020,
}

model = lgb.train(params, train_set=train_matrix, num_boost_round=3300,
                  verbose_eval=1000,
                  valid_sets=valid_matrix,
                  early_stopping_rounds=200)
val_pre_lgb = model.predict(X_test, num_iteration=model.best_iteration)

# plot_roc(y_val, val_pre_lgb)
pd.DataFrame(val_pre_lgb).to_csv('out/result.csv')

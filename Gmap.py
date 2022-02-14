from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow import keras


# 读 json 数据
def get_data(df):
    map_id_list = []
    key_frame_list = []
    status = []
    gap_time = []
    img_name = []
    label = []

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


def model_d(train_x, train_y, test_x):
    length = train_x.shape[0]

    val_x = train_x[int(0.8 * length):]
    val_y = train_y[int(0.8 * length):]

    train_x = train_x[:int(0.8 * length)]
    train_y = train_y[:int(0.8 * length)]

    model = models.Sequential()
    # (6892,4)
    model.add(layers.Dense(32, activation='relu', input_shape=(4,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    callbacks = [keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)]

    history = model.fit(train_x, train_y, epochs=10, shuffle=True, batch_size=128, steps_per_epoch=1024,
                        callbacks=callbacks)

    acc = history.history['acc']
    loss = history.history['loss']

    epoches = range(1, len(acc) + 1)

    plt.plot(epoches, acc, 'b', color='red', label='training acc')
    plt.plot(epoches, loss, 'b', color='blue', label='training loss')
    plt.legend()

    plt.show()

    val = model.evaluate(val_x, val_y, steps=10)
    print('val : {}'.format(val))

    pre = model.predict(test_x)

    return pre


# 数据归一化
def pre_data(train_x, test_x):
    mean = train_x.mean(axis=0)
    train_x -= mean
    std = train_x.std(axis=0)
    train_x /= std

    test_x -= mean
    test_x /= std

    return train_x, test_x


# yolo 产生的结果文件
train_img_x = get_img_data('yolo_out/out_train.csv')
test_img_x = get_img_data('yolo_out/out_28.csv')

# json和图片数据
path = 'C:/Users/Dooooooooo21/Desktop/project/Gmap/'
train_json = pd.read_json(path + 'amap_traffic_annotations_train.json')
test_json = pd.read_json(path + 'amap_traffic_annotations_b_test_0828.json')
train_df = get_data(train_json[:])
test_df = get_data(test_json[:])

# 两个来源数据合并完的 df
train_concat_x = pd.merge(train_img_x, train_df, on=['map_id', 'img_name'])
test_concat_x = pd.merge(test_img_x, test_df, on=['map_id', 'img_name'])

# 选择数据列
select_features = ['counts', 'my_dis', 'hour', 'dayofweek']
train_x = train_concat_x[select_features].copy()
# 将 y 转 onehot 编码
train_y = tf.one_hot(train_concat_x["label"], 3)

test_x = test_concat_x[select_features].copy()

train_x = pd.DataFrame(train_x, dtype=np.float)
test_x = pd.DataFrame(test_x, dtype=np.float)

# 数据预处理-归一化
train_x, test_x = pre_data(train_x, test_x)

sub = test_df[["map_id", "img_name"]].copy()

# 训练模型，预测结果
pre = model_d(train_x, train_y, test_x)

sub["pred"] = np.argmax(pre, axis=1)
sub['pred'].astype(int)

deal_sub = sub.groupby('map_id').agg({'pred': ['mean']}).reset_index()
deal_sub.columns = ['map_id', 'pred']
deal_sub['pred'] = deal_sub['pred'].apply(round)

result_dic = deal_sub.set_index('map_id')['pred'].to_dict()

# 保存
import json

with open(path + "amap_traffic_annotations_b_test_0828.json", "r") as f:
    content = f.read()
content = json.loads(content)
for i in content["annotations"]:
    i['status'] = result_dic[i["id"]]
with open(path + "out.json", "w") as f:
    f.write(json.dumps(content))

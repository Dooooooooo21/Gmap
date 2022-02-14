from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16, ResNet50V2
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, shutil


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


# json和图片数据
path = 'C:/Users/Dooooooooo21/Desktop/project/Gmap/'
train_json = pd.read_json(path + 'amap_traffic_annotations_train.json')
test_json = pd.read_json(path + 'amap_traffic_annotations_test.json')
train_df = get_data(train_json[:])
test_df = get_data(test_json[:])


# 输出结果
def out_result(path, result_dic):
    with open(path + "amap_traffic_annotations_test.json", "r") as f:
        content = f.read()
    content = json.loads(content)
    for i in content["annotations"]:
        i['status'] = result_dic[i["id"]]
    with open(path + "out.json", "w") as f:
        f.write(json.dumps(content))


# 将图片根据类别复制到相应目录
def copy_train_files(df, src_dir, dst_dir):
    rows = len(df)

    for i in range(rows):
        shutil.copyfile(path + src_dir + '/' + str(df.iloc[i]['map_id']) + '/' + df.iloc[i]['img_name'],
                        path + dst_dir + '/' + str(df.iloc[i]['label']) + '/' + str(df.iloc[i]['map_id']) + '_' +
                        df.iloc[i]['img_name'])


# copy test数据
def copy_test_files(df, src_dir, dst_dir):
    rows = len(df)

    for i in range(rows):
        shutil.copyfile(path + src_dir + '/' + str(df.iloc[i]['map_id']) + '/' + df.iloc[i]['img_name'],
                        path + dst_dir + '/' + str(df.iloc[i]['map_id']) + '_' +
                        df.iloc[i]['img_name'])


# 查看shape
def get_shape(gen):
    for data_batch, labels_batch in gen:
        print(data_batch.shape)
        print(labels_batch.shape)
        break


def my_model():
    conv_base = ResNet50V2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), classes=3)
    conv_base.trainable = True
    # set_trainable = False
    # for layer in conv_base.layers:
    #     if layer.name == 'block5_conv1':
    #         set_trainable = True
    #     if set_trainable:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = False

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# copy_train_files(train_df, 'amap_traffic_train_0712', 'train_0712')
# copy_test_files(test_df, 'amap_traffic_test_0712', 'test_0712')

path = 'C:/Users/Dooooooooo21/Desktop/project/Gmap/'


def data_generator():
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=30, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(path + 'train_0712', batch_size=16)
    val_generator = val_datagen.flow_from_directory(path + 'val_0712', batch_size=16)
    test_generator = test_datagen.flow_from_directory(path + 'test_0712', batch_size=16)
    test_generator.reset()

    return train_generator, val_generator, test_generator


# 训练，保存最优模型
def train(train_g, val_g):
    model = my_model()

    # keras.callbacks.EarlyStopping(monitor='acc', patience=1),
    callbacks = [keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0),
                 keras.callbacks.ModelCheckpoint(filepath='model/gmap_10.h5', monitor='val_loss', save_best_only=True)]
    history = model.fit_generator(train_g, steps_per_epoch=200, epochs=80, validation_data=val_g,
                                  validation_steps=20, callbacks=callbacks)


def load_model():
    model_fine = models.load_model('model/gmap_10.h5')
    return model_fine


def pre_test(model, train_g, test_g):
    # 预测
    pred = model.predict_generator(test_g, verbose=1)
    pre_class_indices = np.argmax(pred, axis=1)

    # 处理预测结果
    labels = train_g.class_indices
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in pre_class_indices]

    files = test_g.filenames
    result = pd.DataFrame({'file': files, 'pre': predictions})
    result.to_csv('result_epochs_10_t_13_00.csv')


train_generator, val_generator, test_generator = data_generator()

train(train_generator, val_generator)
model_b = load_model()
pre_test(model_b, train_generator, test_generator)

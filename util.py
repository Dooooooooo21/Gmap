import os, shutil
import pandas as pd
import json


def remove(value):
    value = value.replace('100\\', '')
    return value


# 图像 id
def get_id(value):
    value = value.split('_')[0]
    return value


# 文件名
def get_file(value):
    value = value.split('_')[1]
    return value


# 文件处理
def read_csv(file):
    df = pd.read_csv(file)
    df = df[["file", "pre"]].copy()
    df['file'] = df['file'].apply(remove)
    df['id'] = df['file'].apply(get_id)
    df['img'] = df['file'].apply(get_file)
    deal_sub = df.groupby('id').agg({'pre': ['mean']}).reset_index()
    deal_sub.columns = ['map_id', 'pred']
    deal_sub['pred'] = deal_sub['pred'].apply(round)
    print(deal_sub.head())

    return deal_sub


# 输出
def out_result(dic):
    with open(path + "amap_traffic_annotations_test.json", "r") as f:
        content = f.read()
    content = json.loads(content)
    for i in content["annotations"]:
        i['status'] = dic[i["id"]]
    with open(path + "out.json", "w") as f:
        f.write(json.dumps(content))


path = 'C:/Users/Dooooooooo21/Desktop/project/Gmap/'
result_df = read_csv('out/result_epochs_10_t_10_30.csv')
result_dic = result_df.set_index('map_id')['pred'].to_dict()
out_result(result_dic)
print(result_dic)

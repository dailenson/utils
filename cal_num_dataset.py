'''
Author: your name
Date: 2020-09-15 14:56:07
LastEditTime: 2020-09-15 16:25:12
LastEditors: Please set LastEditors
Description: 计算数据类别的数量
FilePath: /utils/cal_num_dataset.py
'''

import numpy as np
import os

def get_labels(data_path):
    labels_list = []
    with open(data_path,'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            _, label = [i for i in lines.split()]
            labels_list.append(label)
    return labels_list

#适用于labels从0开始的数据集，若从labels从1开始需修改cal_num()
def cal_num(labels_list):
    classes = np.unique(labels_list) ###计算labels的种类数量
    class_total = [0 for i in classes]
    for i in range(len(labels_list)):
        label = int(labels_list[i]) - 1
        class_total[label] += 1 #统计对应label的数量
    return class_total

if __name__ == "__main__":
    for file_name in os.listdir("./data/pacs"):
        path = os.path.join("./data/pacs", file_name)
        labels_list = get_labels(path)
        result = cal_num(labels_list)
        print(f"{file_name} num is {result}")
    

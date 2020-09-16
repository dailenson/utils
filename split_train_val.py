from random import sample, random
from cal_num_dataset import cal_num
import numpy as np
import os

'''
description: 获取图片路径和labels
param {type} 
return {type} 
'''
def get_images_labels(data_path):
    images_list, labels_list = [], []
    with open(data_path,'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            image_path, label = [i for i in lines.split()]
            images_list.append(image_path)
            labels_list.append(int(label))
    return images_list, labels_list

def write_file(images_path, labels, write_path):
    with open(write_path,'w') as file_to_write:
        assert len(images_path) == len(labels), "数据长度和标签长度不相等"
        for i in range(len(images_path)):
            content = images_path[i] + " " + str(labels[i])
            file_to_write.writelines(content + "\n")

def train_test_split(images_list, labels_list, test_size=0.2):
    assert len(images_list) == len(labels_list), "数据长度和标签长度不相等"
    total_number = cal_num(labels_list)
    l = 0
    for i in range(len(total_number)):
        l += total_number[i]
        total_number[i] = l
    images_class = np.split(images_list, total_number)
    labels_class = np.split(labels_list, total_number)
    X_train, X_val, y_train, y_val = [], [], [], []
    for i in range(len(images_class)):  ### 去掉[]
        length = int(len(images_class[i])*test_size)
        np.random.shuffle(images_class[i]) ###shuffling数据顺序
        x_val_path = images_class[i][0:length]
        x_train_path = images_class[i][length:]
        y_val_label = labels_class[i][0:length]
        y_train_label = labels_class[i][length:]
        
        assert len(x_train_path) == len(y_train_label), f"类别{i}train数据长度和标签不相等"
        assert len(x_val_path) == len(y_val_label), f"类别{i}val数据长度和标签不相等"
        X_train.append(x_train_path)
        X_val.append(x_val_path)
        y_train.append(y_train_label)
        y_val.append(y_val_label)
    return np.concatenate(X_train), np.concatenate(X_val), np.concatenate(y_train), np.concatenate(y_val)

if __name__ == "__main__":
    source_path = "./data/office_home"
    traget_path = "./data/office"
    for file_name in os.listdir(source_path):
        images_list, labels_list =  get_images_labels(os.path.join(source_path, file_name))
        X_train, X_val, y_train, y_val = train_test_split(images_list, labels_list, test_size=0.2)
        ### 写入train集
        train_name = os.path.splitext(file_name)[0] + "_train.txt"
        write_file(X_train, y_train, os.path.join(traget_path, train_name))
        ### 写入val集
        val_name = os.path.splitext(file_name)[0] + "_val.txt"
        write_file(X_val, y_val, os.path.join(traget_path, val_name))

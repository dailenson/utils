'''
Author: daigang
Date: 2020-09-14 16:27:52
LastEditTime: 2020-09-15 10:29:32
LastEditors: Please set LastEditors
Description: 计算均值和方差
FilePath: /utils/cal_mean_std.py
'''
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms

transform_data = transforms.Compose([
    #transforms.Resize(224),
    transforms.ToTensor(),
])


pacs = ["art_painting.txt", "cartoon.txt", "photo.txt", "sketch.txt"]
office_home = ["Art.txt", "Clipart.txt", "Product.txt", "RealWorld.txt"]

data_path = "./data"
image_h = 224
image_w = 224

means = []
stds = []
'''
description: 获取txt路径
param {type} 
return {type} 
'''
def get_file_list(dataset_name):
    result_path = []
    for name in pacs:
        temp_path = os.path.join(data_path, name)
        result_path.append(temp_path)
    return result_path

'''
description: 获取数据路径
param {type} 
return {type} 
'''
def get_file_name_list(file_path):
    file_name_list = []
    for temp_path in file_path:
        with open(temp_path,'r') as file_to_read:
                while True:
                    lines = file_to_read.readline()
                    if not lines:
                        break
                    file_name, _ = [i for i in lines.split()]
                    file_name_list.append(file_name)
                    print(file_name)
    return file_name_list

def get_image(image_path):
    img = Image.open(image_path).convert('RGB')
    #img = transform_data(img)
    arr = np.array(img)
    return arr

def cal_mean_std(get_file_name_list):
    image_list = []
    for data in get_file_name_list:
        img = get_image(data)
        img = img[:, :, :, np.newaxis]
        image_list.append(img)
    imgs = np.concatenate(image_list, 3)
    for i in range(3):
        pixels = imgs[i, :, : , :].ravel() #拉成一行
        means.append(np.mean(pixels))
        stds.append(np.std(pixels))
    print("means is ",means)
    print("stds is ", stds) 

if __name__ == "__main__":
       path_list = get_file_list(pacs)
       file_name_list = get_file_name_list(path_list)
       cal_mean_std(file_name_list)

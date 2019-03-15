# coding:utf-8
#!/usr/bin/env python3
# created by xiaqunfeng

import json
import random
import os

# 判断任务类型
def judge_task_type(lib_file):
    task_type = 'unknow'
    with open(lib_file, 'r') as f:
        first_line = f.readline()
        img_info = json.loads(first_line.strip())
        label_name = img_info['label'][0]['name']
        label_type = img_info['label'][0]['type']
        task_type = label_type
    return task_type

# 将沙子库中信息保存为字典
# key：图片名
# value：分类（分类标签）, 检测（data数组）
def save_library_dict(lib_file, task_type):
    lib_dict = {}
    with open(lib_file, 'r') as f:
        for line in f:
            img_info = json.loads(line.strip())
            img_name = img_info['url'].split('/')[-1]
            if task_type == 'classification':
                label = img_info['label'][0]['data'][0]['class']
                lib_dict[img_name] = label
            elif task_type == 'detection' or task_type == 'face':
                data = img_info['label'][0]['data']
                lib_dict[img_name] = data
    return lib_dict

# 给保留信息的沙子随机标签
def stochastic_class(lib_list_i):
    pulp_labels = ['pulp', 'sexy', 'normal']
    terror_labels = ['not terror', 'islamic flag', 'isis flag', 'tibetan flag', 'knives_true', \
            'knives_false', 'knives_kitchen', 'guns_true', 'guns_anime', 'guns_tools']

    lib_list_i_info = json.loads(lib_list_i)
    label_name = lib_list_i_info['label'][0]['name']
    if label_name == 'pulp':
        lib_list_i_info['label'][0]['data'][0]['class'] = random.choice(pulp_labels)
    elif label_name == 'detect':
        bbox_num = len(lib_list_i_info['label'][0]['data'])
        for i in range(bbox_num):
            lib_list_i_info['label'][0]['data'][i]['class'] = random.choice(terror_labels)
    
    sand_list_i = json.dumps(lib_list_i_info)
    return sand_list_i

# 直接从沙子库中抽取指定沙子
def extr_sand_from_library_directly(lib_file, sand_num):
    sand_list = []
    with open(lib_file, 'r') as f:
        lib_list = f.readlines()
        random.shuffle(lib_list)
        for i in range(sand_num):
            sand_list_i = stochastic_class(lib_list[i].strip())
            sand_list.append(sand_list_i)
    return sand_list

# 移除一条图片json文件的分类和检测框信息,即data信息
def remove_sand_data_info(img_info):
    img_info = json.loads(img_info)
    img_info['label'][0]['data'] = []
    new_img_info = json.dumps(img_info)
    return new_img_info

# 抽取沙子，同时将所抽取沙子里的class信息或bbox信息去除
def extr_sand_and_remove_cls_bbox(lib_file, sand_num):
    sand_list = []
    with open(lib_file, 'r') as f:
        lib_list = f.readlines()
        random.shuffle(lib_list)
        for i in range(sand_num):
            img_info = lib_list[i].strip()
            new_img_info = remove_sand_data_info(img_info)
            sand_list.append(new_img_info.strip())
    return sand_list

# 计算两个框的iou
# bbox: [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
def cal_iou(bbox_a, bbox_b):
    Reframe = [bbox_a[0][0], bbox_a[0][1], bbox_a[2][0], bbox_a[2][1]]
    GTframe = [bbox_b[0][0], bbox_b[0][1], bbox_b[2][0], bbox_b[2][1]]
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1, x2+width2)
    startx = min(x1, x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1, y2+height2)
    starty = min(y1, y2)
    height = height1+height2-(endy-starty)

    if width <= 0 or height <= 0:
        ratio = 0  # 重叠率为 0
    else:
        Area = width*height  # 两矩形相交面积
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    return ratio

# 找出已标注框中和沙子框最匹配的
# 如果标签不匹配，直接跳过，标签匹配的话，得到iou最高的那个
# 新增了人脸检测+方向的任务，同样适用，只不过json字段不一样：检测是class，人脸方向是roll_cls
def get_bestmatch_bbox(sand_data_i, labeled_data):
    if 'roll_cls' not in sand_data_i:
        sand_data_i_class = sand_data_i['class']
    else:
        sand_data_i_class = sand_data_i['roll_cls']
    sand_data_i_bbox = sand_data_i['bbox']
    max_iou, max_idx = 0.0, -1
    for i in range(len(labeled_data)):
        if 'roll_cls' not in labeled_data[i]:
            labeled_data_i_class = labeled_data[i]['class']
        else:
            labeled_data_i_class = labeled_data[i]['roll_cls']
        if not (sand_data_i_class == labeled_data_i_class):
            continue
        labeled_data_i_bbox = labeled_data[i]['bbox']
        iou_ratio = cal_iou(labeled_data_i_bbox, sand_data_i_bbox)
        if iou_ratio > max_iou:
            max_iou = iou_ratio
            max_idx = i
    return max_idx, max_iou

# 得到目录下的json文件列表，如果指向是单个文件，则返回该文件
def get_jsonfiles(json_dir):
    jsonfiles = []
    if os.path.isdir(json_dir):
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                jsonfiles.append(filename)
    else: 
        jsonfiles = [json_dir]
        json_dir = ''
    return jsonfiles, json_dir

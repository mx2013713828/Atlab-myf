# coding:utf-8
#!/usr/bin/env python3
# created by xiaqunfeng

import os
import sys
import json
import argparse
import random
from sand_utils import *
from labelx_api_utils import token_gen, get_pkg_info, pass_or_reject_pkg

# 向文件夹下每个pkg插入沙子
def insert_sand_in_pkg(dir_name, lib_file, sand_num, reserve_flag):
    jsonfiles, dir_name = get_jsonfiles(dir_name)
    for json_file in jsonfiles:
        res_lst = []
        filename = os.path.join(dir_name, json_file)
        if reserve_flag:
            sand = extr_sand_from_library_directly(lib_file, sand_num)
        else:
            sand = extr_sand_and_remove_cls_bbox(lib_file, sand_num)
        print(filename)
        with open(filename, 'r') as f:
            for line in f:
                res_lst.append(line.strip())
        res_lst.extend(sand)
        random.shuffle(res_lst)

        with open(filename, 'w') as f:
            f.write('\n'.join(res_lst) + '\n') # 在文件末尾加入换行符，方便cat拼接文件
        print("Successfully INSERT %d sand into package %s!" % (sand_num, json_file))

# 计算分类的pkg准确率
def calculate_cls_acc(dir_name, lib_dict, pass_score):
    pass_pkg_lst, reject_pkg_lst = [], []
    for json_file in os.listdir(dir_name):
        if not json_file.endswith('.json'):
            continue
        filename = os.path.join(dir_name, json_file)
        correct_sand_num, all_sand_num = 0.0, 0.0
        with open(filename, 'r') as f:
            for line in f:
                img_info = json.loads(line.strip())
                img_name = img_info['url'].split('/')[-1]
                if img_info['label'] == [] or img_info['label'][0]['data'] == []:
                    continue
                label = img_info['label'][0]['data'][0]['class']
                if img_name in lib_dict:
                    all_sand_num = all_sand_num + 1
                    if lib_dict[img_name] == label:
                        correct_sand_num = correct_sand_num + 1
        if all_sand_num == 0.0:
            acc = 0.0
        else:
            acc = correct_sand_num / all_sand_num
        print("Pakage %s ACC is: %.3f" % (json_file, acc))
        
        pkg_name = json_file.split('.')[0]
        if acc >= pass_score:
            pass_pkg_lst.append(pkg_name)
        else:
            reject_pkg_lst.append(pkg_name)
    return pass_pkg_lst, reject_pkg_lst

# 计算暴恐检测的pkg准确率
def calculate_det_acc(dir_name, lib_dict, pass_score):
    pass_pkg_lst, reject_pkg_lst = [], []
    for json_file in os.listdir(dir_name):
        if not json_file.endswith('.json'):
            continue
        filename = os.path.join(dir_name, json_file)
        correct_sand_bbox_num, all_sand_bbox_num = 0.0, 0.0
        correct_labeled_bbox_num, all_labeled_bbox_num = 0.0, 0.0
        with open(filename, 'r') as f:
            for line in f:
                img_info = json.loads(line.strip())
                img_name = img_info['url'].split('/')[-1]
                labeled_data = img_info['label'][0]['data']
                if img_name in lib_dict:
                    sand_data = lib_dict[img_name]
                    all_sand_bbox_num += len(sand_data)
                    all_labeled_bbox_num += len(labeled_data)
                    if len(labeled_data) == 0 or len(sand_data) == 0:
                        continue
                    idx_set = {-1}
                    for i in range(len(sand_data)):
                        max_idx, max_iou = get_bestmatch_bbox(sand_data[i], labeled_data)
                        if max_iou >= 0.7 and (max_idx not in idx_set):
                            correct_sand_bbox_num += 1
                            idx_set.add(max_idx)
                    idx_set = {-1}
                    for i in range(len(labeled_data)):
                        max_idx, max_iou = get_bestmatch_bbox(labeled_data[i], sand_data)
                        if max_iou >= 0.7 and (max_idx not in idx_set):
                            correct_labeled_bbox_num += 1
                            idx_set.add(max_idx)
        if all_sand_bbox_num == 0 or all_labeled_bbox_num == 0:
            recall = 0.0
        else:
            recall = correct_sand_bbox_num / all_sand_bbox_num
        if all_labeled_bbox_num == 0:
            precision = 0.0
        else:
            precision = correct_labeled_bbox_num / all_labeled_bbox_num
        print("Pakage %s recall is: %.3f, precision is: %.3f" % (json_file, recall, precision))

        pkg_name = json_file.split('.')[0]
        if recall >= pass_score and precision >= pass_score:
            pass_pkg_lst.append(pkg_name)
        else:
            reject_pkg_lst.append(pkg_name)
    return pass_pkg_lst, reject_pkg_lst

# 计算指定文件夹下每个pkg的准确率
def calculate_acc(dir_name, lib_dict, task_type, pass_score):
    if task_type == 'classification':
        return calculate_cls_acc(dir_name, lib_dict, pass_score)
    elif task_type == 'detection' or task_type == 'face':
        return calculate_det_acc(dir_name, lib_dict, pass_score)

# 根据合格与否的list来执行相应的操作
# 合格则通过审核，不合格直接打回
def pass_and_reject_pkgs(pass_pkg_lst, reject_pkg_lst, pkg_lst_file):
    dict_pkg = {}
    if pkg_lst_file is not None:
        with open(pkg_lst_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.endswith('.json'):
                    continue
                package = line.split(',')[0].split('.')[0] 
                user = line.split(',')[1]
                dict_pkg[package] = user

    token = token_gen()
    if len(pass_pkg_lst) == 0:
        print("\nPASS package list is empty")
    else:
        print("\nPASS package list (num=%d):" % len(pass_pkg_lst))
        for pkg_name in pass_pkg_lst:
            pkg_id, status = get_pkg_info(pkg_name, token)
            if status == 'processed':
                pass_or_reject_pkg(pkg_id, token, True)
            print('\t' + pkg_name)
    if len(reject_pkg_lst) == 0:
        print("\nREJECT package list is empty")
    else:
        print("\nREJECT package list (num=%d):" % len(reject_pkg_lst))
        for pkg_name in reject_pkg_lst:
            pkg_id, status = get_pkg_info(pkg_name, token)
            if status == 'processed':
                pass_or_reject_pkg(pkg_id, token)
            if (pkg_lst_file is not None) and (pkg_name in dict_pkg):
                print('\t' + pkg_name + '\t' + dict_pkg[pkg_name])
            else:
                print('\t' + pkg_name)

# 移除pkg中的沙子
def del_sand_from_labeled_pkg(dir_name, lib_dict):
    for json_file in os.listdir(dir_name):
        if not json_file.endswith('.json'):
            continue
        filename = os.path.join(dir_name, json_file)
        del_sand_num = 0
        res_lst = []
        with open(filename, 'r') as f:
            for line in f:
                try:
                    img_info = json.loads(line.strip())
                except:
                    print(line)
                try:
                    img_name = img_info['url'].split('/')[-1]
                except:
                    #print(filename)
                    print('error',img_info)
                    continue
                if img_name in lib_dict:
                    del_sand_num = del_sand_num + 1
                    continue
                res_lst.append(line.strip())

        with open(filename, 'w') as f:
            f.write('\n'.join(res_lst) + '\n') # 在文件末尾加入换行符，方便cat拼接文件
        print("Successfully DELETE %d sand from package %s!" % (del_sand_num, json_file))
        
if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dir', type=str, help='dir of json package, or single jsonfile')
    ap.add_argument('lib', type=str, help='sand library file')

    ap.add_argument('-n', '--sandnum', type=int,
                    help='Specify the num of sand to insert the package')
    ap.add_argument('-r', '--reserve', action="store_true",
                    help='wether reserve class or bbox info when insert sand')
    ap.add_argument('-v', '--verify', action='store_true')
    ap.add_argument('-l', '--pkglst', type=str,
                    help='package list file of package dir')
    ap.add_argument('-p', '--pass_score', type=float,
                    help='pass score for package. if not set, default score is 0.9')

    group = ap.add_mutually_exclusive_group()
    group.add_argument("-c", "--calculate", action="store_true")
    group.add_argument("-s", "--sand", action="store_true")
    group.add_argument("-d", "--delete", action="store_true")

    args = ap.parse_args()

    usage = """
    Usage:
    python pkg_handle.py json_dir sand_library -s -n sand_num [-r]
    python pkg_handle.py json_dir sand_library -c [-v] [-p score] [-l]
    python pkg_handle.py json_dir sand_library -d
    """
    
    if not (args.lib).endswith('.json'):
        print("Sand file is not json!!")
        os._exit()
    task_type = judge_task_type(args.lib)
    if task_type != 'classification' and task_type != 'detection' and task_type != 'face':
        print("Task type is: %s. Currently not supported!" % task_type)
        os._exit()

    lib_dict = save_library_dict(args.lib, task_type)

    if args.sand and args.sandnum:
        insert_sand_in_pkg(args.dir, args.lib, args.sandnum, args.reserve)
    elif args.calculate:
        if args.pass_score:
            pass_pkg_lst, reject_pkg_lst = calculate_acc(args.dir, lib_dict, task_type, args.pass_score)
        else:
            pass_pkg_lst, reject_pkg_lst = calculate_acc(args.dir, lib_dict, task_type, 0.90)
        if args.verify:
            pass_and_reject_pkgs(pass_pkg_lst, reject_pkg_lst, args.pkglst)
    elif args.delete:
        del_sand_from_labeled_pkg(args.dir, lib_dict)
    else:
        print(usage)

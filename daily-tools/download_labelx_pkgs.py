import argparse
import json
import os
from labelx_api_utils import *

# 打印listname 及其成员
def print_lst(pkg_lst, lst_name):
    if len(pkg_lst) == 0:
        print("%s is empty" % lst_name)
    else:
        print("%s (num=%d):" % (lst_name, len(pkg_lst)))
        for i in pkg_lst:
            print('\t' + i)

# 将包名列表文件读入，得到一个list，便于后续处理
def get_pack_lst(file):
    pack_lst = []
    dict_pkg_user = {}
    with open(file, 'r') as f:
        for line in f:
            pack_info = line.strip()
            if len(pack_info.split(',')) == 2:
                pack_name = pack_info.split(',')[0]
                pack_user = pack_info.split(',')[1]
                pack_lst.append(pack_name)
                dict_pkg_user[pack_name.split('.')[0]] = pack_user
            elif pack_info.endswith('.json'):
                pack_lst.append(pack_info)
    return pack_lst, dict_pkg_user

# 统计所有包的状态并归类，选择是否下载"已提交"状态的包
def count_and_download_pkgs(pkg_lst_file, output_path, download_flag=False, force_flag=False):
    pkg_lst, dict_pkg_user = get_pack_lst(pkg_lst_file)
    if len(pkg_lst) == 0:
        print('Package list file have no invalid list!')
        return 1

    token=token_gen()
    download_failed_lst = []
    assigned_pkg_lst, processed_pkg_lst, reviewed_pkg_lst = [], [], []

    for pkg_name in pkg_lst:
        if download_flag:
            output_json = os.path.join(output_path, pkg_name)
        pkg_name = pkg_name.split('.')[0]
        pkg_id, status = get_pkg_info(pkg_name, token)
        #print("package id = %d status = %s" % (pkg_id, status))
        if force_flag:
            status = 'processed'
        if status == 'processed':
            print('start downloading %s'%pkg_name)
            if pkg_name in dict_pkg_user: 
                processed_pkg_lst.append(pkg_name + '\t' + dict_pkg_user[pkg_name])
            else:
                processed_pkg_lst.append(pkg_name)

            if download_flag:
                if not download_pkg(pkg_id, token, output_json):
                    download_failed_lst.append(pkg_name)
                    print("Downlad %s failed.." % pkg_name)
        elif status == 'assigned':
            assigned_pkg_lst.append(pkg_name)
        elif status == 'reviewed':
            reviewed_pkg_lst.append(pkg_name)
        else:
            print("package %s status is: %s" % (pkg_name, status))

    if not force_flag:
        print_lst(assigned_pkg_lst, 'ASSIGNED package list')
        print_lst(processed_pkg_lst, 'PROCCESSED package list')
        print_lst(reviewed_pkg_lst, 'REVIEWED package list')
    if download_flag:
        print_lst(download_failed_lst, 'DOWNLOAD FAILED list')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="jugg_det toolkit",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('pkg_lst', type=str, help="package name list")
    parser.add_argument('-o', '--output_dir', required=False, type=str,
                        help='output dir for download json package')
    parser.add_argument('-f', '--force', action="store_true",
                        help='force to download packages and ingnore packages status')
    args = parser.parse_args()

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not args.force:
            count_and_download_pkgs(args.pkg_lst, args.output_dir, True)
        else:
            count_and_download_pkgs(args.pkg_lst, args.output_dir, True, True)
    else:
        count_and_download_pkgs(args.pkg_lst, '')

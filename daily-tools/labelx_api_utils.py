# coding:utf-8
# created by xiaqunfeng
import requests
import json
import argparse

# 线上环境
HOST = 'http://labelx-prod.apistore.qiniu.com'
PRE_API = '/api/v2'

def token_gen():
    # 线上环境
    body = json.dumps({"email": "labelx@qiniu.com", "password": "Bob_dylon"})
    r = requests.post(HOST + PRE_API + '/users/signin',data=body, headers={"Content-Type": "application/json"})
    r_json = r.json()
    token = r_json['token']
    return token

def get_assigneeID(file):
    list_uid = []
    list_name = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            username = line.split(',')[0]
            password = line.split(',')[1]
            body = json.dumps({"email": username, "password": password})
            r = requests.post(HOST + PRE_API + '/users/signin',data=body, headers={"Content-Type": "application/json"})
            try:
                r_json = r.json()
            except:
                print(line,r)
            uid = r_json['id']
            list_uid.append(int(uid))
            list_name.append(username)
    return list_uid, list_name

def create_dataset_request(dict_dataset):
    body=json.dumps(dict_dataset)
    token=token_gen()
    try:
        r = requests.post(HOST + PRE_API + '/datasets',timeout=5, data=body, headers={"Content-Type": "application/json", "Authorization": token})
        if r.status_code != 200:
            print(r)
            return -1
        #print(r.text)
        pkg_id = json.loads(r.text)['id']
        return pkg_id
    except Exception as e:
        print("http Exception: %s"%e)
        return -2

def get_pkg_info(pkg_name, token):
    try:
        r = requests.get(HOST + PRE_API + '/datasets?title=%s&limit=100'%pkg_name, headers={"Authorization": token})
        rj = json.loads(r.text)['items']
        #print(r.text)
        pkg_id = -1
        status = 'initial'
        for i in range(len(rj)):
            if rj[i]['title'] == pkg_name:
                pkg_id = rj[i]['id']
                status = rj[i]['status']
                break
        return pkg_id, status
    except Exception as e:
        print("Http Exception: %s" % e)
        print("Get package %s info faild!" % pkg_name)

def get_pkglist_by_keyword(pkg_name, token):
    try:
        r = requests.get(HOST + PRE_API + '/datasets?&title=%s&limit=20&offset=0&mode=&'%pkg_name, headers={"Authorization": token})
        rj = json.loads(r.text)['items']
        #print(r.text)
        pkg_id = -1
        status = 'initial'
        for i in range(len(rj)):
            pkg_realname = rj[i]['title']
            pkg_id = rj[i]['id']
            status = rj[i]['status']
            assignees = rj[i]['assignees'][0]['email']
            pic_num = rj[i]['stats']['fileCount']
            create_time = rj[i]['createTime']
            print("pkg_name: %-35s status: %-10s account: %-25s pic_num: %-6d create_time: %-13s pkg_id: %d" % (pkg_realname, status, assignees, pic_num, create_time.split('T')[0], pkg_id))
    except Exception as e:
        print("Http Exception: %s" % e)
        print("Get packages info by keyword [ %s ] faild!" % pkg_name)
        
def download_pkg(pkg_id, token, output_json):
    success_flag = False
    try:
        json_file = requests.post(HOST + PRE_API + '/datasets/%d/export'%(pkg_id), timeout=5,headers={"Content-Type": "application/json", "Authorization": token})
        json_items = json_file.json()
        content = '\n'.join(json.dumps(line) for line in json_items)
        with open(output_json, 'w') as f:
            f.write(content)
        success_flag = True
    except Exception as e:
        print("Http Exception: %s" % e)
    return success_flag

def pass_or_reject_pkg(pkg_id, token, pass_flag=False):
    if pass_flag:
        body = json.dumps({"status": "reviewed"})
    else:
        body = json.dumps({"status": "assigned"})
    try:
        r = requests.post(HOST + PRE_API + '/datasets/%d'%(pkg_id), data=body, timeout=5,headers={"Content-Type": "application/json", "Authorization": token})
        #print(r.text)
    except Exception as e:
        print(r)
        print("Http Exception: %s" % e)

if __name__=='__main__':
    ap = argparse.ArgumentParser(description="get package info by keyword")    
    ap.add_argument('keyword', type=str, help='keyword for search package in labelx')
    args = ap.parse_args()

    token = token_gen()
    get_pkglist_by_keyword(args.keyword, token)

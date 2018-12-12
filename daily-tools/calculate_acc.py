import json,os,argparse


def _init_():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='dir of json package, or single jsonfile')
    parser.add_argument('lib', type=str, help='sand library file')
    args = parser.parse_args()

    return args
def calculate_cls_acc(dir_name, lib_dict, pass_score=0.9):
    pass_pkg_lst, reject_pkg_lst = [], []
    for json_file in os.listdir(dir_name):
        if not json_file.endswith('.json'):
            continue
        if json_file.endswith('-error.json'):
            continue
        filename = os.path.join(dir_name, json_file)
        correct_sand_num, all_sand_num = 0.0, 0.0
        error_sand = []
        with open(filename, 'r') as f:
            error_file = json_file.split('.')[0]+'-error'+'.json'
            errorfile = os.path.join(dir_name,error_file)
            
            for line in f:
                img_info = json.loads(line.strip())
                img_name = img_info['url'].split('/')[-1]
                if img_info['label'] == [] or img_info['label'][0]['data'] == []:
                    continue
                label = img_info['label'][0]['data'][0]['class']

                if img_name in lib_dict:
                    all_sand_num = all_sand_num + 1
                    # print(lib_dict[img_name],label)
                    if lib_dict[img_name] == label:
                        # print(lib_dict[img_name],label)
                        correct_sand_num = correct_sand_num + 1
                    else:
                        img_info["label"][0]["data"].append({'"class"':lib_dict[img_name]})
                        line = '{"url":"%s","type":"image","label":[{"name":"general","type":"classification","version":"1","data":[{"class":"%s"},{"class":"%s"}]}]}\n'%(img_info['url'], label,lib_dict[img_name])

                        error_sand.append(line)
        print(all_sand_num)
        with open(errorfile,'w') as f:
            for line in error_sand:
                f.write(line)
        if all_sand_num == 0.0:
            acc = 0.0
        else:
            acc = float(correct_sand_num) / float(all_sand_num)
        print("Pakage %s ACC is: %.3f" % (json_file, acc))

        pkg_name = json_file.split('.')[0]
        if acc >= pass_score:
            pass_pkg_lst.append(pkg_name)
        else:
            reject_pkg_lst.append(pkg_name)
    return pass_pkg_lst, reject_pkg_lst

if __name__ == "__main__":
    args = _init_()
    lib_dict = dict()
    with open(args.lib,'r') as f :
        for line in f:
            img_info = json.loads(line.strip())
            img_name = img_info['url'].split('/')[-1]
            label = img_info['label'][0]['data'][0]['class']

            lib_dict[img_name] = label
            
    pass_pkg_lst, reject_pkg_lst = calculate_cls_acc(args.dir, lib_dict, pass_score=0.9)
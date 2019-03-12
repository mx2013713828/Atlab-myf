#coding:utf-8
#用于更新沙子，再将之前标错的沙子中gt错误的更新之后，对现有的沙子库进行更新，注意url唯一性
import json,os,argparse
def _init_():
    parser = argparse.ArgumentParser()
    parser.add_argument('newsand', type=str, help='changed sand')
    parser.add_argument('oldsand', type=str, help='old sand')
    # parser.add_argument('--output', type=str, help='output save file ')
    args = parser.parse_args()
    return args


    
def main():
    num = 0
    args = _init_()
    print(args)
    allsands = {}
    with open(args.oldsand) as f:
        for line in f:
            js = json.loads(line.strip())
            label = js['label'][0]['data'][0]['class']
            url = js['url']
            allsands[url] = label
    for root,_,files in os.walk(args.newsand):
        for file in files:
            print(file)
            newsand = os.path.join(root,file)
            if newsand.endswith('.DS_Store'):
                continue
            with open(newsand) as f:
                for line in f:
                    js = json.loads(line.strip())
                    url = js['url']
                    try:
                        if len(js['label'][0]['data']) ==2:
                            label = js['label'][0]['data'][1]['class']
                        else:
                            label = js['label'][0]['data'][0]['class']
                    except:
                        print(js)

                    try:
                        if label!=allsands[url]:
                            num+=1
                    except:
                        num+=1
                    
                    allsands[url] = label
                print('num',num)
    with open(args.oldsand,'w') as f:
        for key in allsands:
            label = allsands[key]
            line = '{"url":"%s","type":"image","label":[{"name":"general","type":"classification","version":"1","data":[{"class":"%s"}]}]}\n'%(key, label)
            f.write(line)


if __name__ == "__main__":
    print('---start----')
    main()

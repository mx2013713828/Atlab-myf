#coding:utf-8
import json,os,argparse,random

def _init_():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='dir of json package, or single jsonfile')
    parser.add_argument('lib', type=str, help='sand library file')
    parser.add_argument('--normal',type=str,help='normal num')
    parser.add_argument('--sexy', type=str, help='sexy num')
    parser.add_argument('--pulp', type=str, help='pulp num')
    args = parser.parse_args()
    return args
def main(): 
    args = _init_()
    sands = []
    # newres = []
    labelx = ['generic_normal','ani_porn','ani_sexy','generic_sexy','flirt_sex_con']
    with open(args.lib) as f:
        for line in f:
            sands.append(line.strip())
    for root,_,files in os.walk(args.dir):
        
        for file in files:
            p = 0
            s = 0
            newres = []
            filename = os.path.join(root,file)
            if filename.endswith('.DS_Store'):
                continue
            print(filename+',inserting sands')
            with open(filename,'r') as f:
                #shuffle沙子
                random.shuffle(sands)
                for line in f:
                    newres.append(line.strip())

                for line in sands:
                    # print(line)
                    js = json.loads(line)
                    try:
                        label = js['label'][0]['data'][0]['class']
                    except Exception as e:
                        print(e)
                        label = 'generic_porn'
                    url = js['url']
                    if label.endswith('porn') and p < int(args.pulp):
                        lb = random.choice(labelx)
                        line = '{"url":"%s","type":"image","label":[{"name":"general","type":"classification","version":"1","data":[{"class": "%s"}]}]}'%(url,lb)

                        newres.append(line)
                        p +=1
                    elif s<int(args.sexy):
                        lb = random.choice(labelx)
                        line = '{"url":"%s","type":"image","label":[{"name":"general","type":"classification","version":"1","data":[{"class": "%s"}]}]}'%(url,lb)

                        s+=1
                        newres.append(line)
            #加入沙子之后的shuffle *very important*
            random.shuffle(newres)
            with open(filename,'w') as f:
                for line in newres:
                    f.write(line + '\n')
            print('successful insert %s pulp -and- %s sexy'%(p,s))


if __name__ == "__main__":
    main()
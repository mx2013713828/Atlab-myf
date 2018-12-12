import json,os,argparse,random

def _init_():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='dir of json package, or single jsonfile')
    parser.add_argument('lib', type=str, help='sand library file')
    parser.add_argument('--sexy', type=str, help='sexy num')
    parser.add_argument('--pulp', type=str, help='pulp num')
    args = parser.parse_args()
    return args
def main(): 
    args = _init_()
    sands = []
    # newres = []
    
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
            print(filename)
            with open(filename,'r') as f:
                random.shuffle(sands)
                for line in f:
                    newres.append(line.strip())
            # random.shuffle(sands)
                for line in sands:
                    js = json.loads(line)
                    label = js['label'][0]['data'][0]['class']
                    url = js['url']
                    if label.endswith('porn') and p < int(args.pulp):
                        line = '{"url":"%s","type":"image","label":[{"name":"general","type":"classification","version":"1","data":[]}]}'%(url)

                        newres.append(line)
                        p +=1
                    elif not label.endswith('porn') and s<int(args.sexy):
                        line = '{"url":"%s","type":"image","label":[{"name":"general","type":"classification","version":"1","data":[]}]}'%(url)
                        
                        s+=1
                        newres.append(line)
            with open(filename,'w') as f:
                for line in newres:
                    f.write(line + '\n')
                print('successful insert %s pulp -and- %s sexy'%(p,s))




            
    
if __name__ == "__main__":
    main()
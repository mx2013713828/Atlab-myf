import json
import argparse,os

def _init_():
    parser = argparse.ArgumentParser()
    parser.add_argument('labelx', type=str, help='dir of json package, or single jsonfile')
    args = parser.parse_args()
    return args

    
def main():
    args = _init_()
    alx = list()
    with open(args.labelx) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)

            if len(js['label'][0]['data'])==1:
                label = js['label'][0]['data'][0]['class']
            elif len(js['label'][0]['data'])==2:
                label = js['label'][0]['data'][1]['class']
            url = js['url']
            line = '{"url":"%s","type":"image","label":[{"name":"general","type":"classification","version":"1","data":[{"class":"%s"}]}]}\n'%(url, label)
            alx.append(line)
    with open(args.labelx,'w') as f:
        for line in alx:
            f.write(line)

if __name__ == "__main__":
    main()
    
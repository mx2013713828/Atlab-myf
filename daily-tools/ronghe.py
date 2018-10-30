import os,json,argparse

parser = argparse.ArgumentParser()
parser.add_argument('fileA', type=str, help='dir_a')
parser.add_argument('fileB', type=str, help='dir_b')
parser.add_argument('output', type=str, help='output save file ')
args = parser.parse_args()

input_a = args.fileA
input_b = args.fileB

pkg_a = []
pkg_b = []

for root,_,files in os.walk(input_a):
    print('root',root)
    for filename in files:
        if filename == '.DS_Store':
            os.remove(os.path.join(root,filename))
            continue
        filename = os.path.join(root,filename)
        print(filename)
        with open(filename) as f:
            for row in f:
                pkg_a.append(row.strip())
for root,_,files in os.walk(input_b):
    print('root',root)
    for filename in files:
        filename = os.path.join(root,filename)
        print(filename)
        with open(filename) as f:
            for row in f:
                pkg_b.append(row.strip())

set_a = set(pkg_a)
set_b = set(pkg_b)

dic_a = {}
for row in set_a:
    js = json.loads(row)
    url = js['url'].strip()
    try:
        label = js['label'][0]['data'][0]['class'].strip()
    except:
        print(js)
    dic_a[url]=label

diff_same = list()
diff = list()
for row in set_b:
    try:
        js = json.loads(row)
        url = js['url'].strip()
    except:
        print(row)
    try:
        label = js['label'][0]['data'][0]['class'].strip()
        if label == dic_a[url]:
            diff_same.append(row)
        else:
            diff.append(row)

        
    except:
        pass
print('benefit dataset num:',len(diff_same))

with open(args.output+'.json','w') as f:
    for row in diff_same:
        f.write(row+'\n')

with open(args.output+'_diff.json','w') as f:
    for row in diff:
        f.write(row + '\n')


# if __name__ == '__main__':
#     main()


import os
with open('log.lst','w') as f:
    for i in range(1,10):
        f.write("qpulp_origin_2019010"+str(i)+".json")
        f.write('\n')
with open('log.lst','a') as f:
    for i in range(10,30):
        f.write("qpulp_origin_201901"+str(i)+".json")
        f.write('\n')
with open('log.lst','a') as f:
    for i in range(1,10):
        f.write("qpulp_origin_2019020"+str(i)+".json")
        f.write('\n')
       
    
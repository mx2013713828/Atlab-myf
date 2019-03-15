# -*- coding: UTF-8 -*-
import threading
from time import sleep,ctime
#coding:utf-8
import cv2
import numpy as np
from math import *
import random
import os


#
def flip(img):
    x = random.randint(0,2)
    flipped = cv2.flip(img,x)
    return flipped

def rotate(img):

    cap = random.randint(0,90)
    (h,w) = img.shape[:2]
    center = (w / 2,h / 2)

    M = cv2.getRotationMatrix2D(center,cap,1)#旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
    rotated = cv2.warpAffine(img,M,(w,h))
    return rotated
def shift(img):
    x = random.randint(-200,200)


    M = np.float32([[1,0,x],[0,1,x]])
    shifted = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    return shifted

def mergeA(img1,img2):
    #resize tuya
    img1 = cv2.resize(img1,(img2.shape[1],img2.shape[0]),interpolation=cv2.INTER_AREA)
    img3 = img2
    ws = []
    hs = []
#     print(img1.shape)
#     print(img2.shape)
    for i in range(int(img1.shape[0]-1)):
        for j in range(int(img1.shape[1]-1)):
            a = img1[i][j]
            b = img2[i][j]
#             if ([0,0,0]==a).all():
#                 img3 [i][j] = b
            if a[2]>160:
                ws.append(j)
                hs.append(i)

    #             print(i,j)

                img3[i][j] = a
    return img3,ws,hs


def mergeB(img1,img2):
    #resize scene
    img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]),interpolation=cv2.INTER_AREA)
    img3 = img2
    ws = []
    hs = []
#     print(img1.shape)
#     print(img2.shape)
    for i in range(int(img1.shape[0]-1)):
        for j in range(int(img1.shape[1]-1)):
            a = img1[i][j]
            b = img2[i][j]
#             if ([0,0,0]==a).all():
#                 img3 [i][j] = b
            if a[2]>160:
                ws.append(j)
                hs.append(i)

    #             print(i,j)

                img3[i][j] = a
    return img3,ws,hs


class myThread (threading.Thread):
    def __init__(self, threadID, name, s , e):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.s = s
        self.e = e
    def run(self):
        print ("Starting " + self.name+ctime())
        # 获得锁，成功获得锁定后返回True
        # 可选的timeout参数不填时将一直阻塞直到获得锁定
        # 否则超时后将返回False
        threadLock.acquire()
        #线程需要执行的方法
        printImg(self.s,self.e)
        # 释放锁
        threadLock.release()
listImg = [] #创建需要读取的列表，可以自行创建自己的列表
for i in range(179):
    listImg.append(i)
# 按照分配的区间，读取列表内容，需要其他功能在这个方法里设置
fw = open('anno-multi.lst','w')
def printImg(s,e):
#     for i in range(int(s),int(e)):
#         print (i)
#     fw = open('anno-4.lst','w')
    for i,scene in enumerate(scenes[int(s):int(e)]):
        sce = cv2.imread(scene)
        modle = random.sample(mods,1)[0]

        try:
            mod = cv2.imread(modle)
            if i%11 ==0:
                mod = shift(mod)

            elif i%5 ==0:
                mod = flip(mod)
            elif i %7 ==0:
                mod = rotate(mod)

            merge1,ws1,hs1 = mergeA(mod,sce)
            x1 = min(ws1)
            y1 = min(hs1)
            x2 = max(ws1)
            y2 = max(hs1)
            w = merge1.shape[1]
            h = merge1.shape[0]
            filename = '/workspace/mnt/group/general-reg/mayufeng/tuya/train2018/tuya_v1_'+scene.split('/')[-1]
            cv2.imwrite(filename,merge1)
            fw.write(filename+',' +str(h)+','+str(w)+','+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+'\n')

            merge2,ws2,hs2 = mergeB(mod,sce)
            x1 = min(ws2)
            y1 = min(hs2)
            x2 = max(ws2)
            y2 = max(hs2)
            w = merge2.shape[0]
            h = merge2.shape[1]
            filename = '/workspace/mnt/group/general-reg/mayufeng/tuya/train2018/tuya_v2_'+scene.split('/')[-1]
            cv2.imwrite(filename,merge2)
            fw.write(filename+',' +str(h)+','+str(w)+','+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+'\n')

            fw.flush()
        except Exception as e: 
            print(e)
            print(scene)
            print(modle)
            
#             mod = cv2.imread(modle)
#             merge1,ws1,hs1 = mergeA(mod,sce)
#             x1 = min(ws1)
#             y1 = min(hs1)
#             x2 = max(ws1)
#             y2 = max(hs1)
#             w = merge1.shape[1]
#             h = merge1.shape[0]
#             filename = '/workspace/mnt/group/general-reg/mayufeng/tuya/train2018/tuya_v1_'+scene.split('/')[-1]
#             cv2.imwrite(filename,merge1)
#             fw.write(filename+',' +str(h)+','+str(w)+','+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+'\n')

#             merge2,ws2,hs2 = mergeB(mod,sce)
#             x1 = min(ws2)
#             y1 = min(hs2)
#             x2 = max(ws2)
#             y2 = max(hs2)
#             w = merge2.shape[0]
#             h = merge2.shape[1]
#             filename = '/workspace/mnt/group/general-reg/mayufeng/tuya/train2018/tuya_v2_'+scene.split('/')[-1]
#             cv2.imwrite(filename,merge2)
#             fw.write(filename+',' +str(h)+','+str(w)+','+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+'\n')

#             fw.flush()
        
        
mods = list()
scenes = list()
with open('mods.lst') as f:
    for row in f:
        filename = row.strip()
        mods.append(filename)
with open('scenemod-v2.lst') as f:
    for row in f:
        filename = row.strip()
        scenes.append(filename)
# scenes = scenes[15000:20000]
    
totalThread = 16 #需要创建的线程数，可以控制线程的数量
lenList = len(scenes) #列表的总长度
gap = lenList / totalThread #列表分配到每个线程的执行数
threadLock = threading.Lock() #锁
threads = [] #创建线程列表
# 创建新线程和添加线程到列表
for i in range(totalThread):
    thread = 'thread%s' % i
    if i == 0:
        thread = myThread(0, "Thread-%s" % i, 0,gap)
    elif totalThread==i+1:
        thread = myThread(i, "Thread-%s" % i, i*gap,lenList)
    else:
        thread = myThread(i, "Thread-%s" % i, i*gap,(i+1)*gap)
    threads.append(thread) # 添加线程到列表
# 循环开启线程
for i in range(totalThread):
    threads[i].start()
# 等待所有线程完成
for t in threads:
    t.join()
print ("Exiting Main Thread")
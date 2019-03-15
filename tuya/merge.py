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
    color = [random.randint(1,254),random.randint(1,254),random.randint(1,254)]
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

                img3[i][j] = [4,248,10]
    return img3,ws,hs


def mergeB(img1,img2):
    #resize scene
    img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]),interpolation=cv2.INTER_AREA)
    img3 = img2
    ws = []
    hs = []
#     print(img1.shape)
#     print(img2.shape)
    color = [random.randint(1,254),random.randint(1,254),random.randint(1,254)]
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

                img3[i][j] = [229,250,243]
    return img3,ws,hs


mods = list()
scenes = list()
with open('mods.lst') as f:
    for row in f:
        filename = row.strip()
        mods.append(filename)
with open('scenemod.lst') as f:
    for row in f:
        filename = row.strip()
        scenes.append(filename)

fw = open('anno-colors-7-v1v2.lst','w')
for i,scene in enumerate(scenes[44000:45000]):
    sce = cv2.imread(scene)
    modle = random.sample(mods,1)[0]

    try:
        if i%2 ==0:
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
        else:
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
#         mod = cv2.imread(modle)
#         merge1,ws1,hs1 = mergeA(mod,sce)
#         x1 = min(ws1)
#         y1 = min(hs1)
#         x2 = max(ws1)
#         y2 = max(ws1)
#         w = merge1.shape[1]
#         h = merge1.shape[0]
#         filename = '/workspace/mnt/group/general-reg/mayufeng/tuya/train2018/tuya_v1_'+scene.split('/')[-1]
#         cv2.imwrite(filename,merge1)
#         fw.write(filename+',' +str(h)+','+str(w)+','+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+'\n')

#         merge2,ws2,hs2 = mergeB(mod,sce)
#         x1 = min(ws2)
#         y1 = min(hs2)
#         x2 = max(ws2)
#         y2 = max(ws2)
#         w = merge2.shape[0]
#         h = merge2.shape[1]
#         filename = '/workspace/mnt/group/general-reg/mayufeng/tuya/train2018/tuya_v2_'+scene.split('/')[-1]
#         cv2.imwrite(filename,merge2)
#         fw.write(filename+',' +str(h)+','+str(w)+','+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+'\n')

#         fw.flush()
    
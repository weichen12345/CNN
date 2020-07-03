# -*- coding: utf-8 -*-
# import MySQLdb
import pymysql
from shutil import copyfile
import os
import cv2
import numpy as np
from codecs import open
import time

img_path = '/home/rsd/image/image_geetest'
train_path = '/home/rsd/image/image_geetest_train'
# train_path = os.path.join(os.getcwd(),'train_new')
print(train_path)

db = pymysql.connect(host="172.16.0.76",
                     user="fengyuanhua",
                     passwd= "!@#qweASD",
                     db="geetest_image",
                     charset='utf8')

cur = db.cursor()
cur.execute("select name, coordinate, characters, midpoint from geetest_image.geetest_image where status = 3")
num = 0
# temp = open('temp.txt','w',encoding='utf-8')
for row in cur.fetchall():
    (f, coors, chars, points) = row
    if points == '': continue
    prefix, ext = os.path.splitext(f)
    fpath = os.path.join(img_path, f)
    coors = [int(cr) for cr in coors[1:-1].strip().split(',')]
    points = [int(p) for p in points[1:-1].split(',')]
    if len(chars) != len(points) / 2: continue
    if not os.path.exists(fpath):
        print(f)
        # temp.write(f)
        num +=1
        print(num)
    else:
        img = cv2.imread(fpath)
        for idxp in range(0, len(points), 2):
            ch = chars[int(idxp/2)]
            px1,py1 = points[idxp:idxp+2]
            # for idxc in range(0, len(coors), 4):
            #     x1,y1,x2,y2 = coors[idxc:idxc+4]
            #     if x1 < px1 < x2 and y1 < py1 < y2:
            #         if not os.path.exists(os.path.join(train_path, ch)):
            #             os.mkdir(os.path.join(train_path, ch))
            #         p = train_path + '\\' + ch + '\\' + prefix + '_' + str(idxp) + '.jpg'
            #         print(p)
            #         cv2.imencode('.jpg', img[y1:y2, x1:x2])[1].tofile(p)
            x1 = px1 - 32
            x2 = px1 + 32
            y1 = py1 - 32
            y2 = py1 + 32
            if x1 < 0:
                x1 = 0
            if x2 > 344:
                x2 = 344
            if y1 < 0:
                y1 = 0
            if y2 > 344:
                y2 = 344
            if not os.path.exists(os.path.join(train_path, ch)):
                os.mkdir(os.path.join(train_path, ch))
            p = train_path + '/' + ch + '/' + prefix + '_' + str(idxp) + '.jpg'
                # print(p)
            cv2.imencode('.jpg', img[y1:y2, x1:x2])[1].tofile(p)
# temp.close()
#
# cur.close()

# cur = db.cursor()
# cur.execute("select name, coordinate, characters, midpoint from geetest_image.geetest_image where status = 1")
#
# for row in cur.fetchall():
#     (f, coors, chars, points) = row
#     prefix, ext = os.path.splitext(f)
#     fpath = os.path.join(img_path, f)
#     if not os.path.exists(fpath):
#         print(f)
#     else:
#         img = cv2.imread(fpath)
#         coors = [int(cr) for cr in coors[1:-1].strip().split(', ')]
#         if len(chars) != len(coors) / 4: continue
#         for idxc in range(0, len(coors), 4):
#             ch = chars[int(idxc/4)]
#             x1,y1,x2,y2 = coors[int(idxc):int(idxc)+4]
#             if not os.path.exists(os.path.join(train_path, ch)):
#                 os.mkdir(os.path.join(train_path, ch))
#             # cv2.imwrite(os.path.join(train_path, ch, prefix + '_' + str(idxc) + '.jpg'), img[y1:y2, x1:x2])
#             p = train_path + '/' + ch + '/' + prefix + '_' + str(idxc) + '.jpg'
#             # print(p)
#             # time.sleep(100)
#             cv2.imencode('.jpg', img[y1:y2, x1:x2])[1].tofile(p)

    
cur.close()
db.close()




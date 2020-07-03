# coding=utf8
import cv2
import numpy as np
import test_2
import matplotlib.pyplot as plt

def cut_pic(path):
    img = cv2.imread(path)
    img1 = img[0:112,0:112]
    img2 = img[0:112,116:228]
    img3 = img[0:112,232:344]
    img4 = img[116:228,0:112]
    img5 = img[116:228,116:228]
    img6 = img[116:228,232:344]
    img7 = img[232:344,0:112]
    img8 = img[232:344,116:228]
    img9 = img[232:344,232:344]
    # a = cv2.resize(img,[112,112])
    # image = img9.resize([112, 112,3])
    imgs = [img1,img2,img3,img4,img5,img6,img7,img8,img9]
    i=1
    for _img in imgs:
        # image = _img.resize([112,112])
        image = np.array(_img)
        test_2.evaluate_one_image(image,i)
        i+=1



if __name__ == '__main__':
    path = r'C:\Users\14686\Desktop\test\test.jpg'
    cut_pic(path)

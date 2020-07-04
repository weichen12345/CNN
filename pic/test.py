# -*- coding: utf-8 -*-

import os
import json
from codecs import open
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential, load_model
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.utils import multi_gpu_model
from keras.preprocessing import image
import  numpy as np
import time
# from picchar.train import VGG
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


PATCH_SIZE = 64
NB_CLASSES = 3473
initializer1 = 'ones'
# initializer = 'truncated_normal'
initializer = 'truncated_normal'
def VGG(weights_path=None):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(PATCH_SIZE, PATCH_SIZE, 3),
                     kernel_initializer=initializer1))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same", kernel_initializer=initializer1))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same", kernel_initializer=initializer1))
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same", kernel_initializer=initializer1))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(256, (3, 3), activation='relu', padding="same", kernel_initializer=initializer1))
    model.add(Conv2D(256, (3, 3), activation='relu', padding="same", kernel_initializer=initializer1))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(512, (3, 3), activation='relu', padding="same", kernel_initializer=initializer))
    model.add(Conv2D(512, (3, 3), activation='relu', padding="same", kernel_initializer=initializer))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)
    return model



# model = load_model('.\\model\\model-90.h5')
model = load_model('./model/model-20.h5')
# model = VGG()
# model.load_weights('./model/model-78.h5')
# model = multi_gpu_model(model, gpus=2)
# model = load_model('.\\model\\model-old.h5')
imgList = os.listdir('./test')
# with open('data-239.json','r',encoding='utf8') as f:
with open('./data/data.json','r',encoding='utf8') as f:
# with open('.\data\data-old.json','r',encoding='utf8') as f:
    r = f.readline()
    res = json.loads(r)
    # print(res)
wordDict = {v:k for k,v in res.items()}
# print(wordDict)
# time.sleep(1000)
accNum = 0

for num, imgName in enumerate(imgList):
    imgPath = './test/' + imgName
    print(imgPath)
    # img = image.load_img(imgPath, target_size=(64,64),grayscale=True)
    img = image.load_img(imgPath, target_size=(64,64),grayscale=False)
    imgArray = image.img_to_array(img)
    imgInput = np.expand_dims(imgArray, 0)
    # print(imgName[0])
    res = model.predict(imgInput)
    resList = list(res[0])
    # print(num)
    # resList1 = sorted(resList,reverse=True)
    # print(resList1)
    # print(res)

    b = sorted(enumerate(resList), key=lambda x: x[1], reverse=True)
    print(b)
    wordIndex = b[0][0]
    # wordPred = wordDict[wordIndex]
    # print(wordPred)
    # if wordPred == imgName.split('_')[0]:
    print(wordIndex)
    print(imgName)
    if wordIndex == int(imgName.split('_')[0]):
        accNum += 1
    print('accuracy: ',accNum/(num+1))
    print('@'*100)
    # time.sleep(1000)




# imgList = os.listdir('/home/rsd/image/train_word_new/丑/')
# for num, imgName in enumerate(imgList):
#     imgPath = '/home/rsd/image/train_word_new/丑/' + imgName
#     # print(imgPath)
#     # img = image.load_img(imgPath, target_size=(64,64),grayscale=True)
#     img = image.load_img(imgPath, target_size=(64,64),grayscale=False)
#     imgArray = image.img_to_array(img)
#     imgInput = np.expand_dims(imgArray, 0)
#     res = model.predict(imgInput)
#     resList = list(res[0])
#     print(resList)
#     # print(num)
#     # resList1 = sorted(resList,reverse=True)
#     # print(resList1)
#     # print(res)
#
#     b = sorted(enumerate(resList), key=lambda x: x[1], reverse=True)
#     # print(b)
#     wordIndex = b[0][0]
#     wordPred = wordDict[wordIndex]
#     print(wordPred)
#
#




# C:\code\image\image_word
#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import  regularizers
import json
from codecs import open
from keras.applications.vgg16 import VGG16
import numpy as np

def VGG(weights_path=None):
    model = Sequential()
    # model.add(Conv2D(128, (3, 3), activation='relu', padding="same", input_shape=(PATCH_SIZE, PATCH_SIZE, 3)))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same",  input_shape=(PATCH_SIZE, PATCH_SIZE, 3)))
    # model.add(Conv2D(64, (3, 3), activation='relu', padding="same",name='conv2d_1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))
    # model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(512, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    # model.add(Dense(2048, activation='relu',kernel_regularizer=regularizers.l1(0.01),activity_regularizer=regularizers.l1(0.01)))
    model.add(Dense(256, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    
    if weights_path:
        model.load_weights(weights_path)
    return model

PATCH_SIZE = 64
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam(lr=0.001)
NB_CLASSES = 143
# NB_CLASSES = 3405
# INPUT_SHAPE = (PATCH_SIZE, PATCH_SIZE, 1)
# ROOT_PATH = "C:\code\captfr\picchar"
ROOT_PATH = os.getcwd()
MODEL_DIR = os.path.join(ROOT_PATH, "model")
# MODEL_DIR = "D:\picmodel"
TRAIN_DIR = os.path.join(ROOT_PATH, "train")
# TRAIN_DIR = os.path.join(ROOT_PATH, "image_geetest_train")
TRAIN_CNT = 11540 -1094
# TRAIN_CNT = 129288

# keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
#     samplewise_center=False,
#     featurewise_std_normalization=False,
#     samplewise_std_normalization=False,
#     zca_whitening=False,
#     zca_epsilon=1e-6,
#     rotation_range=0.,
#     width_shift_range=0.,
#     height_shift_range=0.,
#     shear_range=0.,
#     zoom_range=0.,
#     channel_shift_range=0.,
#     fill_mode='nearest',
#     cval=0.,
#     horizontal_flip=False,
#     vertical_flip=False,
#     rescale=None,
#     preprocessing_function=None,
#     data_format=K.image_data_format())

train_datagen = ImageDataGenerator(
    # shear_range=30.,
    width_shift_range=0.05,
    height_shift_range=0.05,
    # rotation_range=10.,
    zoom_range=[0.95,1.05],
    rescale=1./255
    )
train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64,64),
        class_mode='categorical',
        batch_size=BATCH_SIZE,
    # color_mode='grayscale'
)
dic = train_generator.class_indices
print(train_generator.class_indices)
# time.sleep(1000)
with open(os.path.join(os.getcwd(), "data",'data.json'),'w',encoding='utf-8') as f:
    a = json.dumps(dic,ensure_ascii=False)
    f.write(a)
print(train_generator.classes)
print(train_generator.directory)
print(train_generator.class_mode)
# model = VGG('D:\picmodel\model-446.h5')
# modelOld = load_model('C:\code\captfr\picchar\model\model-old.h5')

# print('models layers:',modelOld.layers)
# print('models config:',modelOld.get_config())
# print('models summary:',modelOld.summary())
model = VGG()
# model.load_weights('C:\code\captfr\picchar\model\model-old.h5',by_name=True)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
model.summary()
# checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"))
checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "model.h5"))
model.fit_generator(
        train_generator, 
        steps_per_epoch=TRAIN_CNT // BATCH_SIZE, 
        epochs=NB_EPOCH,
        verbose=VERBOSE, 
        callbacks=[checkpoint])


# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *
from keras.models import load_model
import os
import json
import sys
from codecs import open
sys.setrecursionlimit(40000)
np.set_printoptions(threshold=np.inf)


def update_img():
    datagen = ImageDataGenerator(
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-6,
            rotation_range=30.,
            width_shift_range=0.,
            height_shift_range=0.,
            brightness_range=None,
            shear_range=0.,
            zoom_range=0.2,
            channel_shift_range=50.,
            fill_mode='nearest',
            cval=0.,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0)

    img = load_img('C:\code\captfr\picchar\\train\蔼/f45c19b56a49f41cade98de94ae8bd2497a99b6b7216ad23858a398278e5381b0205c515c72a62536110d5ff65d4aa92_4.jpg')  # 这是一个PIL图像
    x = img_to_array(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
    x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)

    # 下面是生产图片的代码
    # 生产的所有图片保存在 `preview/` 目录下
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='C:\code\captfr\picchar\\train\蔼\\test', save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > 10:
            break  # 否则生成器会退出循环

def predict_img():
    file_path = 'C:\code\captfr\picchar\\train\矮\\9eb099677b11b3775b06c816777ae216e762376b4c4defa1c69d6d498155b04a2f33c5c124da6c8c3c52b317dbdc76dd_0.jpg'

    img = image.load_img(file_path, target_size=(64,64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    model1 = load_model('D:/picmodel/model-239.h5')
    model2 = load_model('./model/model_pic.h5')
    # model = ResNet50(weights='imagenet')
    y1 = model1.predict(x)
    y2 = model2.predict(x)
    # r = model.predict_classes(x,verbose=1)
    label = get_label()
    # r = model_pic.predict(b.reshape(1, 64, 64, 3))
    print(np.argsort(y1)[0][-60:])
    print(list(y1[0]))
    for num,i in enumerate(y1[0]):
        if i != 0.0:
            print(num,'  ',i)
    print(list(y1[0][-60:]).sort(reverse=True))
    print(''.join(reversed([label[idx] for idx in np.argsort(y1[0])[-60:]])))
    print(''.join(reversed([label[idx] for idx in np.argsort(y2[0])[-60:]])))
    # print(np.array(y))
    # print(np.argmax(y))
    # print('Predicted:', decode_predictions(y))
def get_label():
    # label_list = os.listdir('C:\code\captfr\picchar\\train')
    with open('data.json','r',encoding='utf-8') as f:
        r =f.readline()
        res = json.loads(r)
        print(res)
    res = {v:k for k,v in res.items()}
    print(res)
    # print(label_list)
    return res
if __name__ == '__main__':
    predict_img()
    # get_label()
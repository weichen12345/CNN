import matplotlib.pyplot as plt
import model_2
from inpute_date_2 import get_files
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import index

# 获取一张图片
def get_one_image(train):
    # 输入参数：train,训练图片的路径
    # 返回参数：image，从训练图片中随机抽取一张图片
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]  # 随机选择测试的图片

    img = Image.open(img_dir)
    plt.imshow(img)
    plt.show()
    image = np.array(img)
    return image


# 测试图片
def evaluate_one_image(image_array,_index):
    label = index.index()
    with tf.Graph().as_default():
        BATCH_SIZE = 1

        N_CLASSES = len(label)

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 112, 112, 3])

        logit = model_2.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[112, 112, 3])

        # you need to change the directories to yours.
        logs_train_dir = r'D:\MyProjects\understand\save_2'
        logs_train_dir = os.path.join(os.getcwd(),'..\\','understand','save_2')
        print(logs_train_dir)
        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            # ckpt = tf.train.get_checkpoint_state(r'D:\MyProjects\understand\save')
            # if ckpt and ckpt.model_checkpoint_path:

            if True:
                global_step = ckpt.model_checkpoint_path.split('\\')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            # else:
            #     print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            for i in range(len(label)):
                if max_index == i:
                    # result = ('这是{}的可能性为：'.format(label[str(i)]) + '%.6f' % prediction[:, i])
                    print(label[str(i)] + '-' + str(_index))

            # return result


# ------------------------------------------------------------------------

if __name__ == '__main__':
    label = index.index()
    img = Image.open(r'D:\MyProjects\inpute_date_2\9_4.jpg')
    plt.imshow(img)
    plt.show()
    imag = img.resize([112, 112])
    image = np.array(imag)
    print(evaluate_one_image(image,0))



import matplotlib.pyplot as plt
import model
from input_data import get_files
from PIL import Image
import numpy as np
import tensorflow as tf
import os
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
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 8

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 112, 112, 3])

        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[112, 112, 3])

        # you need to change the directories to yours.
        logs_train_dir = r'D:\MyProjects\understand\save'
        logs_train_dir =os.path.join(os.getcwd(),'..','understand','save')
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
            if max_index == 0:
                result = ('这是长颈鹿的可能性为： %.6f' % prediction[:, 0])
            elif max_index == 1:
                result = ('这是大象的可能性为： %.6f' % prediction[:, 1])
            elif max_index == 2:
                result = ('这是猴子的可能性为： %.6f' % prediction[:, 2])
            elif max_index == 3:
                result = ('这是生姜的可能性为： %.6f' % prediction[:, 3])
            elif max_index == 4:
                result = ('这是井盖的可能性为： %.6f' % prediction[:, 4])
            elif max_index == 5:
                result = ('这是五角星的可能性为： %.6f' % prediction[:, 5])
            elif max_index == 6:
                result = ('这是乌龟的可能性为： %.6f' % prediction[:, 6])
            else:
                result = ('这是书的可能性为： %.6f' % prediction[:, 7])
            print(prediction)

            return result


# ------------------------------------------------------------------------

if __name__ == '__main__':
    img = Image.open(r'D:\MyProjects\inpute_date\wugui\34.jpg')
    plt.imshow(img)
    plt.show()
    imag = img.resize([112, 112])
    image = np.array(imag)
    print(evaluate_one_image(image))



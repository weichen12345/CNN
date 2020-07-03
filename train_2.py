
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这一行注释掉就是使用cpu，不注释就是使用gpu
import inpute_date_2
import model_2
import tensorflow as tf
import numpy as np
import index
import time
# 变量声明
label = index.index()
N_CLASSES = len(label) # 四种花类型
IMG_W = 112  # resize图像，太大的话训练时间久
IMG_H = 112
BATCH_SIZE = 1024

CAPACITY = 200
MAX_STEP = 15001  # 一般大于10K
learning_rate = 0.0001  # 一般小于0.0001

# 获取批次batch
# train_dir = r'D:\MyProjects\inpute_date_2'  # 训练样本的读入路径
train_dir = os.path.join(os.getcwd(),'input_data_2')  # 训练样本的读入路径
# logs_train_dir = r'D:\MyProjects\understand\save_2'  # logs存储路径
logs_train_dir = os.path.join(os.getcwd(),'understand','save_2')  # logs存储路径
# logs_test_dir = r'D:\PyCharm\KinZhang_First_ImageDetection\generate_data_2'
logs_test_dir = os.path.join(os.getcwd(),'understand','generate_data_2')
# train, train_label = input_data.get_files(train_dir)
train, train_label, val, val_label = inpute_date_2.get_files(train_dir, 0.2)
# print(val)
# 训练数据及标签
train_batch, train_label_batch = inpute_date_2.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
# 测试数据及标签
val_batch, val_label_batch = inpute_date_2.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

# 训练操作定义
train_logits = model_2.inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model_2.losses(train_logits, train_label_batch)
train_op = model_2.trainning(train_loss, learning_rate)
train_acc = model_2.evaluation(train_logits, train_label_batch)

# 测试操作定义
test_logits = model_2.inference(val_batch, BATCH_SIZE, N_CLASSES)
test_loss = model_2.losses(test_logits, val_label_batch)
test_acc = model_2.evaluation(test_logits, val_label_batch)
test_op = model_2.trainning(test_loss,learning_rate)

# 这个是log汇总记录
summary_op = tf.summary.merge_all()

# 产生一个会话
sess = tf.Session()
# 产生一个writer来写log文件
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
val_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)
# 产生一个saver来存储训练好的模型
saver = tf.train.Saver()
# 所有节点初始化
sess.run(tf.global_variables_initializer())
# 队列监控
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 进行batch的训练
try:
    # 执行MAX_STEP步的训练，一步一个batch
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc1 = sess.run([train_op, train_loss, train_acc])
        _, test_loss1, test_acc1 = sess.run([test_op, test_loss, test_acc])

        # 每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
        if step % 1 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc1 * 100.0))
            print('step %d,test loss = %.2f,test accuracy = %.2f%%' % (step, test_loss1, test_acc1 * 100))

            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
        # 每隔100步，保存一次训练好的模型
        if (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()



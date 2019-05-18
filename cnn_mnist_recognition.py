# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.app.flags.DEFINE_boolean("is_train", False, "指定程序是训练还是测试")
FLAGS = tf.app.flags.FLAGS


# 定义一个初始化权重的函数
def weight_varialbles(shape):
    return tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))

# 定义一个初始化偏置的函数
def bias_variables(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))

def model():
    """自定义卷积模型"""

    # 1. 准备数据占位符 x [None, 784] y_true [None, 10]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])


    # 2. 一卷积层 卷积：5*5*1，32个filter, strides=1 激活：tf.nn.relu 池化
    with tf.variable_scope("conv1"):
        # 随机初始化权重，偏置 32
        w_conv1 = weight_varialbles([5,5,1,32])
        b_conv1 = bias_variables([32])

        # 对x进行形状改变[None, 784] [None, 28, 28, 1]
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])

        # [None, 28, 28, 1] ------> [None, 28, 28, 32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1,1,1,1], padding="SAME") + b_conv1)

        # 池化 2*2， strides2 [None, 28, 28, 32]---->[None, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")


    # 3. 二卷积层 卷积：5*5*32 64个filter，strides=1 激活：tf.nn.relu 池化
    with tf.variable_scope("conv2"):
        # 随机初始化权重  权重：[5,5,32，64] 偏置[64]
        w_conv2 = weight_varialbles([5,5,32,64])
        b_conv2 = bias_variables([64])

        # 卷积 激活 池化计算
        # [None, 14, 14, 32]----->[None, 14, 14, 64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1,1,1,1], padding="SAME")+b_conv2)

        # 池化 2*2， strides 2, [None, 14, 14, 64]---->[None, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    # 4. 全连接层 [None,7,7,64]--->[None, 7*7*64]*[7*7*64, 10] + [10] = [None, 10]
    with tf.variable_scope("full_connection"):
        # 随机初始化权重和偏置
        w_fc = weight_varialbles([7*7*64, 1000])
        b_fc = bias_variables([1000])

        # 修改形状[None, 7, 7, 64]--->[None, 7*7*64]
        x_fc_reshape = tf.reshape(x_pool2, [-1,7*7*64])

        # 非线性映射到1000维向量
        x_fc = tf.nn.relu(tf.matmul(x_fc_reshape, w_fc) + b_fc)



    # 5. 输出层
    with tf.variable_scope("Out_result"):
        w_out = weight_varialbles([1000,10])
        b_out = bias_variables([10])

        # 进行矩阵运算得出每个样本的10个结果
        y_predict = tf.matmul(x_fc, w_out) + b_out

    return x, y_true, y_predict


def conv_fc():
    # 1.获取真实数据
    mnist = input_data.read_data_sets("./data/mnist/input_data", one_hot=True)

    # 2.定义模型，得出输出
    x, y_true, y_predict = model()

    # 3.进行交叉熵损失计算
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4.梯度下降求出损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.00005).minimize(loss)

    # 5.计算准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))

        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 创建一个saver
    saver = tf.train.Saver()


    # 开启会话运行
    with tf.Session() as sess:
        sess.run(init_op)

        if FLAGS.is_train:
            # 循环训练
            for i in range(10000):
                # 取出真实存在的特征值和目标值
                mnist_x, mnist_y = mnist.train.next_batch(50)

                # 运行train_op训练
                sess.run(train_op, feed_dict={x:mnist_x, y_true:mnist_y})
                print("训练%d步，准确率为：%f"%(i+1, sess.run(accuracy, feed_dict={x:mnist_x, y_true:mnist_y})))

                # 每训练1000步保存一次模型
                if (i+1)%500 == 0:

                    if not os.path.exists("./model"):
                        os.mkdir("./model")

                    # 保存模型
                    saver.save(sess, "./model/cnn_mnist_recognition")
        else:
            # 加载模型
            saver.restore(sess, "./model/cnn_mnist_recognition")

            true_count = 0
            for i in range(500):
                # 每次测试一张图片
                x_test, y_test = mnist.test.next_batch(1)

                mnist_true = tf.argmax(y_test, 1).eval()
                mnist_predict = tf.argmax(sess.run(y_predict, feed_dict={x:x_test, y_true:y_test}), 1).eval()

                print("第%d次测试: 真实值：%d, 预测值：%d" %(
                    i+1,
                    mnist_true,
                    mnist_predict
                ))

                if mnist_predict == mnist_true:
                    true_count += 1
            acc = true_count/500*100
            print("准确率：%.2f%%" % acc)



if __name__ == '__main__':

    conv_fc()






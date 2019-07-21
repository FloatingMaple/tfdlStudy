# 线性回归实例，目标值是一些输入向量x 的线性组合，给每个样本添加高斯噪音i ，想要找到一组权重w 和偏置项b

import numpy as np
# 使用NumPy生成合成数据。
# 创建2000个分别拥有三个特征的向量样本x
x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2

noise = np.random.randn(1,2000)*0.1
# 将每个x 样本的内积乘以一组权重w ，并添加偏差项b ，最后加上高斯噪音得到结果
y_data = np.matmul(w_real,x_data.T) + b_real + noise

NUM_STEPS = 10

import tensorflow as tf
g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape=None)

    # 使用命名空间进行分层
    # 预测y
    with tf.name_scope('inference') as scope:
        # 以0 初始化 w、b
        w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
        b = tf.Variable(0,dtype=tf.float32,name='bias')
        y_pred = tf.matmul(w,tf.transpose(x)) + b

    # 损失函数
    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true-y_pred))

    # 训练以最小化loss
    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train,{x:x_data,y_true:y_data})
            # if (step%5==0):
            #     print(step,sess.run([w,b]))
            #     wb_.append(sess.run([w,b]))
            print(step,sess.run([w,b]))
            wb_.append(sess.run([w,b]))

        print(10,sess.run([w,b]))
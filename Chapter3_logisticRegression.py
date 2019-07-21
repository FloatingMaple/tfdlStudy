# 一个逻辑回归例子；在逻辑回归框架中，从模拟数据中学习权重和偏置的集合。
# wx+b 称为逻辑函数的非线性函数的输入
# 通过 P = 1/(1+exp{-(wx+b)}) 将线性部分的值压缩到区间[0,1]中，然后视为生成二值 yes/1 或 no/0 输出的概率

import numpy as np

N = 20000
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 创建数据和模拟结果
x_data = np.random.randn(N,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2
wxb = np.matmul(w_real,x_data.T) + b_real

y_data_pre_noise = sigmoid(wxb)
y_data = np.random.binomial(1,y_data_pre_noise)

import tensorflow as tf
# 要使用的loss 是交叉熵的二值版本，也是逻辑回归模型的似然函数：
# y_pred = tf.sigmoid(y_pred)
# loss = y_true*tf.log(y_pred) - (1-y_true)*tf.log(1-y_pred)
# loss = tf.reduce_mean(loss)
# tensorflow中提供可以直接使用的函数：
# tf.nn.sigmoid_cross_entropy_with_logits(labels=,logits=)

NUM_STEPS = 50

g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape=None)

    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
        b = tf.Variable(0,dtype=tf.float32,name='bias')
        y_pred = tf.matmul(w,tf.transpose(x)) + b
        y_pred = tf.sigmoid(y_pred)


    with tf.name_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
        loss = tf.reduce_mean(loss)

    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train,{x:x_data,y_true:y_data})
            if(step%5 == 0):
                print(step,sess.run([w,b]))
                wb_.append(sess.run([w,b]))
        print(50,sess.run([w,b]))
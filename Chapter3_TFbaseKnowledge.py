import tensorflow as tf

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)

# 启动图
sess = tf.Session()
# 完成运行。方法被调用时会完成途中的一组计算：从请求的输出开始，反向进行计算
outs = sess.run(f)
# 完成计算后，关闭会话释放资源
sess.close()
print("outs={}".format(outs))

a1 = tf.constant(5)
b1 = tf.constant(3)

c1 = a1*b1
d1 = a1+b1
e1 = d1-c1
f1 = c1+d1
g1 = f1/e1

sess1 = tf.Session()
outs1 = sess1.run(g1)
sess1.close()
print("outs1={}".format(outs1))

print(tf.get_default_graph)
g = tf.Graph()
print(g)

a2 = tf.constant(5)
print(a.graph is g)
print(a.graph is tf.get_default_graph())


g1 = tf.get_default_graph()
g2 = tf.Graph()
print(g1 is tf.get_default_graph())

with g2.as_default():
    print(g1 is tf.get_default_graph())

print(g1 is tf.get_default_graph())

# 使用with语句打开会话能够保证一旦所有计算完成会话将自动关闭
with tf.Session() as sess2:
    fetches = [a,b,c,d,e,f]
    outs = sess2.run(fetches)

print("outs={}".format(outs))
print(type(outs[0]))

# 该操作创建了一个节点，实际上是一个Tensor对象实例
a2 = tf.constant(5)
b2 = tf.constant(4.0)
print(a2)
print(b2)

c2 = tf.constant(4.0,dtype=tf.float64)
print(c2)
print(c2.dtype)

x = tf.constant([1,2,3],name='x',dtype=tf.float32)
print(x.dtype)
x = tf.cast(x,tf.int64)
print(x.dtype)

import numpy as np
c3 = tf.constant([[1,2,3],
                 [4,5,6]])
print("Python List input: {}".format(c3.get_shape()))

c3 = tf.constant(np.array([
    [[1,2,3]],
    [[1,1,1]]
]))
print("3d Numpy array input: {}".format(c3.get_shape()))

sess = tf.InteractiveSession()
c = tf.linspace(0.0,4.0,5)
print("The content of 'c':\n {} \n".format(c.eval()))
sess.close()

A = tf.constant([
    [1,2,3],
    [4,5,6]
])
print(A.get_shape())
x = tf.constant([1,0,1])
print(x.get_shape())
x2 = tf.transpose(x)
x = tf.expand_dims(x,1)
x1 = tf.constant([[1],[0],[1]])
print(x.get_shape())
print(x1.get_shape())
print(x2.get_shape())
print(x)
print(x1)
print(x2)
b = tf.matmul(A,x)
b1 = tf.matmul(A,x1)
sess = tf.InteractiveSession()
print('matmul result:\n{}'.format(b.eval()))
print('matmul result1:\n{}'.format(b1.eval()))
sess.close()

with tf.Graph().as_default():
    c1 = tf.constant(4,dtype=tf.float64,name='c')
    c2 = tf.constant(4,dtype=tf.int32,name='c')
print(c1.name)
print(c2.name)


with tf.Graph().as_default():
    c1 = tf.constant(4,dtype=tf.float64,name='c')
    with tf.name_scope("prefix_name"):
        c2 = tf.constant(4,dtype=tf.int32,name='c')
        c3 = tf.constant(4,dtype=tf.float64,name='c')

print(c1.name)
print(c2.name)
print(c3.name)

# 随机从服从正态分布的数值中取出指定个数的值
init_val = tf.random_normal((1,5),0,1)
# init_val = tf.random_normal([1,5],0,1)
var = tf.Variable(init_val,name='var')
print("pre run: \n{}".format(var))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    post_var = sess.run(var)

print("\npost run:\n{}".format(post_var))

# ph = tf.placeholder(tf.float32,shape=(None,10))

# sess.run(s,feed_dict={x:X_data,w:w_data})

x_data = np.random.randn(5,10)
w_data = np.random.randn(10,1)

with tf.Graph().as_default():
    # 占位符 x、w
    x = tf.placeholder(tf.float32,shape=(5,10))
    w = tf.placeholder(tf.float32,shape=(10,1))
    # 填充值为-1的常数向量 b
    b = tf.fill((5,1),-1.)
    # x、w矩阵相乘
    xw = tf.matmul(x,w)

    xwb = xw+b
    # 获取该向量的最大值，使用 reduce 将一个5个元素的向量化归到一个标量
    s = tf.reduce_max(xwb)
    with tf.Session() as sess:
        outs = sess.run(s,feed_dict={x:x_data,w:w_data})
        outs1 = sess.run(xwb,feed_dict={x:x_data,w:w_data})

print("outs={}".format(outs))
print("xwb={}".format(outs1))

# f = wx+b  y = f+E 的回归模型
x = tf.placeholder(tf.float32,shape=[None,3])
y_true = tf.placeholder(tf.float32,shape=None)
w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
b = tf.Variable(0,dtype=tf.float32,name='bias')

# 预测输出 y_pred
y_pred = tf.matmul(w,tf.transpose(x)) + b
# 使用均方误差 MSE 作为损失函数
loss = tf.reduce_mean(tf.square(y_true-y_pred))
# 使用交叉熵作为损失函数
loss1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
loss1 = tf.reduce_mean(loss1)

# 使用梯度下降优化器最小化损失函数
# 设置学习率
learning_rate = 0.01
# 创建有指定学习率的优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# 创建一个训练操作，通过调用minimize 来更新变量，并将损失作为参数传递
train = optimizer.minimize(loss)
# 然后通过sess.run() 执行该训练操作
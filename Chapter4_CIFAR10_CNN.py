import tensorflow as tf
import numpy as np
import os
import pickle
# import matplotlib.pyplot as plt

DATA_PATH = "./cifar-10-batches-py"
BATCH_SIZE = 50
STEPS = 500000

# layers 模型中的不同层
def weight_variable(shape):
    # 定义网络层的权重，使用一个标准差为0.1的截断式正态分布初始化
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 指定网络层偏置量，使用0.1进行初始化
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    # 指定卷积操作
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    # 设置最大池化为高度和宽度维度的一半大小，总共是1/4的特征图
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def conv_layer(input,shape):
    # 实际使用的卷积层。线性卷积为conv2d定义，有一个偏置，然后接ReLU非线性函数
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input,W) + b)

def full_layer(input,size):
    # 标准全连接层加上偏置
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size,size])
    b = bias_variable([size])
    return tf.matmul(input,W)+b


# data：图像数据  labels：图像类标
def unpickle(file):
    with open(os.path.join(DATA_PATH,file),'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'latin1'
        dict = u.load()
        return dict

# 从长度为10（0-9的整数）的向量的重编码
def one_hot(vec,vals=10):
    n = len(vec)
    out = np.zeros((n,vals))
    out[range(n),vec] = 1
    return out

# 展示一些图片更好的了解该数据集
def display_cifar(images,size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) for i in range(size)])
    plt.imshow(im)
    plt.show()

class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1,6)])\
            .load()
        self.test = CifarLoader(["test_batch"]).load()

class CifarLoader(object):
    def __init__(self,source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)\
            .astype(float) / 255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]),10)
        return self

    def next_batch(self,batch_size):
        x,y = self.images[self._i:self._i+batch_size],self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x,y


# 展示一些图片更好的了解该数据集
def create_cifar_image():
    d = CifarDataManager()
    print("Number of train images: {}".format(len(d.train.images)))
    print("Number of train labels: {}".format(len(d.train.labels)))
    print("Number of test images: {}".format(len(d.test.images)))
    print("Number of test labels: {}".format(len(d.test.labels)))
    images = d.train.images
    display_cifar(images, 10)

# 简单的 CIFA10 模型
cifar = CifarDataManager()

x = tf.placeholder(tf.float32,shape=[None,32,32,3])
y_ = tf.placeholder(tf.float32,shape=[None,10])
keep_prob = tf.placeholder(tf.float32)

conv1 = conv_layer(x,shape=[5,5,3,32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool,shape=[5,5,32,64])
conv2_pool = max_pool_2x2(conv2)
conv2_flat = tf.reshape(conv2_pool,[-1,8*8*64])

full_1 = tf.nn.relu(full_layer(conv2_flat,1024))
full1_drop = tf.nn.dropout(full_1,keep_prob=keep_prob)

y_conv = full_layer(full1_drop,10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy =  tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

def test(sess):
    X = cifar.test.images.reshape(10,1000,32,32,3)
    Y = cifar.test.labels.reshape(10,1000,10)
    acc = np.mean([sess.run(accuracy,feed_dict={x:X[i],y_:Y[i],keep_prob:1.0})
                   for i in range(10)])
    print("Accuracy: {:.4}%".format(acc*100))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        batch = cifar.train.next_batch(BATCH_SIZE)
        sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

    test(sess)
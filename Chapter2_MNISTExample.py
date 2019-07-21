import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# DATA_DIR = 'D:\Code\MNISTdataset'
DATA_DIR = '.\MNISTdataset'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100


# read_data_sets()将MNIST数据集下载到本地，参数：（路径，读取标签的方式）
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

# 使用占位符 placeholder 和变量 variable 元素构建
# variable 变量是由计算所操控的对象，placeholder 占位符是触发该计算时所需要的对象
# [None,784]表示每张图的维度大小是784（将维度28*28的图像像素展开为一个向量），None表示当前不指定每次使用的图片数量
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

# 学习函数：数据样本到各自相对的已知标签的映射；在本例中是数字图像到图像之出现的数字的映射
# 真实的标签
y_true = tf.placeholder(tf.float32, [None, 10])
# 预测的标签
y_pred = tf.matmul(x, W)

# 使用交叉熵 cross_entropy 测量相似度 (即损失函数 loss function)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_pred, labels=y_true))

# 控制如何训练，使用梯度下降优化法。此处设置学习率为 0.5
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 定义评估步骤，测试模型的准确率。
# correct_mask 测试样本中被正确分类的数据
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
# 计算模型准确率
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# 为了使用已定义的计算图，需要创建一个会话 session
with tf.Session() as sess:

    # Train
    # 初始化所有参数
    sess.run(tf.global_variables_initializer())
    # 训练步骤数 NUM_STEPS
    for _ in range(NUM_STEPS):
        # 在每一步中，向数据管理器申请一定数量的样本以及对应的标签；使用MINIBATCH_SIZE控制每一步使用的样本数量
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        # 使用feed_dict 作为sess.run的参数。构建模型时定义了占位符，执行运算时需要包含这些元素，需要提供数值
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Test
    # 测试模型在测试集上的准确率
    ans = sess.run(accuracy, feed_dict={x: data.test.images,
                                        y_true: data.test.labels})
# 用百分比的格式输出结果
print("Accuracy: {:.4}%".format(ans*100))
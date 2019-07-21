import tensorflow as tf

# 1、引入MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".\MNISTdataset",one_hot=True)

# 1、定义参数
element_size = 28 # 序列中每个向量的维数
time_steps = 28 # 序列中这些元素的数量
num_classes = 10
batch_size = 128
hidden_layer_size = 128

# TensorBoard model summaries
LOG_DIR = ".\logs\RNN_with_summaries"

# 1、创建输入、标签的占位符
_inputs = tf.placeholder(tf.float32,shape=[None,time_steps,element_size],
                                           name='inputs')
y = tf.placeholder(tf.float32,shape=[None,num_classes],
                   name='labels')

#batch_x, batch_y = mnist.train.next_batch(batch_size)
# 2、重塑数据获得28个28元素的序列
#batch_x = batch_x.reshape((batch_size,time_steps,element_size))

# 3、从Tensor flow官方文档得到的函数，简单的添加一些操作保存记录总结
def variable_summaries(var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)
      
# 3、把RNN 步骤中用到的权重和偏差创建为变量
# Weights and bias for 输入和隐藏层
with tf.name_scope('rnn_weights'):
    with tf.name_scope("W_x"):
        Wx = tf.Variable(tf.zeros([element_size,hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope("W_h"):
        Wh = tf.Variable(tf.zeros([hidden_layer_size,hidden_layer_size]))
        variable_summaries(Wh)
    with tf.name_scope("Bias"):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)
        
# 4、用tf.scan() 应用RNN 步骤
def rnn_step(previous_hidden_state,x):
    current_hidden_state = tf.tanh(tf.matmul(previous_hidden_state,Wh)+
                                   tf.matmul(x,Wx) + b_rnn)
    return current_hidden_state

# 4、将该函数应用于所有28个时间步骤(time step)
    
# 通过transpose进行转置，perm不设置时就是基本转置操作，
#perm列表的每一位表示现在的索引轴对应的原来的索引轴。
#如[1,0,2]即原来的1置于0，0置于1，2置于2
    
# 当前输入：（batch_size,timesteps,element_size）
process_input = tf.transpose(_inputs,perm=[1,0,2])
# 当前输入：（time_steps,batch_size,element_size）

initial_hidden = tf.zeros([batch_size,hidden_layer_size])
# 随时间获取所有状态向量
# tf.scan() 将一个可调用的函数按顺序重复作用于元素序列
all_hidden_states = tf.scan(rnn_step,process_input,initializer=initial_hidden,
                            name='states')

# 5、在一个RNN里，每一个时间步骤，都有一个状态向量state vector，将它乘以一定的权重，
#得到一个输出向量，也就是数据的新表征representation。

# 获取输出层权重
# 定义线性层的权重和偏差变量
with tf.name_scope('linear_layer_weight') as scope:
    with tf.name_scope("W_linear"):
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size,num_classes],
                                             mean=0,stddev=.01))
        variable_summaries(Wl)
    with tf.name_scope("Bias_linear"):
        bl = tf.Variable(tf.truncated_normal([num_classes],
                                             mean=0,stddev=.01))
        variable_summaries(bl)

# 将线性层用于状态向量
def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state,Wl)+bl
        
with tf.name_scope('linear_layer_weight') as scope:
    # 根据时间迭代，将线性层用于所有RNN输出
# tf.map_fn 类似经典的map函数，将某些函数应用到序列/可迭代的数据中的每一个元素上
    all_outputs = tf.map_fn(get_linear_layer,all_hidden_states)
    # 获取最后输出
# 用负索引提取此次批处理每一个实例的最后一个输出。
    output = all_outputs[-1]
    tf.summary.histogram('outputs',output)
    
# 6、对分类器classifier进行训练，定义损失函数计算、优化和预测的操作，
#给TensorBoard添加更多摘要，并将摘要合并为一个操作。
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=output,labels=y))
    tf.summary.scalar('cross_entropy',cross_entropy)
    
with tf.name_scope('train'):
    # 使用 RMSPropOptimizer 作为优化器
    train_step = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cross_entropy)
    
with tf.name_scope('accuracy'):
#tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(output,1))
#tf.cast(x,dtype,name=None)tensorflow中张量数据类型转换
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction,tf.float32)))*100
    tf.summary.scalar('accuracy',accuracy)
    
# 合并所有summary
merged = tf.summary.merge_all()
        
# 7、使用一个小测试集进行测试
test_data = mnist.test.images[:batch_size].reshape((-1,time_steps,element_size))
test_label = mnist.test.labels[:batch_size]

with tf.Session() as sess:
    #通过TensorBoard将summaries写入LOG_DIR
    train_writer = tf.summary.FileWriter(LOG_DIR+'/train',
                                         graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR+'/test',
                                        graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    
    for i in range(10000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 sequences of 28 pixels
        batch_x = batch_x.reshape((batch_size, time_steps, element_size))
        summary, _ = sess.run([merged, train_step],
                              feed_dict={_inputs: batch_x, y: batch_y})
        # Add to summaries
        train_writer.add_summary(summary, i)

        if i % 1000 == 0:
            acc, loss, = sess.run([accuracy, cross_entropy],
                                  feed_dict={_inputs: batch_x,
                                             y: batch_y})
            print("Iter " + str(i) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
        if i % 100 == 0:
            # Calculate accuracy for 128 mnist test images and
            # add to summaries
            summary, acc = sess.run([merged, accuracy],
                                    feed_dict={_inputs: test_data,
                                               y: test_label})
            test_writer.add_summary(summary, i)

    test_acc = sess.run(accuracy, feed_dict={_inputs: test_data,
                                             y: test_label})
    print("Test Accuracy:", test_acc)
        
        
        
#        
#        batch_x, batch_y = mnist.train.next_batch(batch_size)
#        #重塑数据获得28个28元素的序列
#        batch_x = batch_x.reshape((batch_size,time_steps,element_size))
#        
#        summary,_ = sess.run([merged,train_step],
#                             feed_dict={_inputs:batch_x,y:batch_y})
#        #添加到summaries
#        train_writer.add_summary(summary,i)
#        
#        if i%1000 == 0 :
#            acc,loss, = sess.run([accuracy,cross_entropy],
#                                 feed_dict={_inputs:batch_x,y:batch_y})
#            print("Iter "+str(i)+", Minibatch Loss= "+"{:.6f}".format(loss)+
#                  ",Training Accuracy= "+"{:.5f}".format(acc))
#        if i % 10 :
#            #计算128MNIST测试图像的准确度accuracy，添加到summaries
#            summary,acc = sess.run([merged,accuracy],
#                                   feed_dict={_inputs:test_data,y:test_label})
#            test_writer.add_summary(summary,i)
#            
#    test_acc = sess.run(accuracy,feed_dict={_inputs:test_data,y:test_label})
#    
#    print("Test Accuracy: ",test_acc)
#        
#        
#        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
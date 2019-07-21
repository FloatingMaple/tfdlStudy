# =============================================================================
# tf.contrib.rnn.BasicRNNCell
# tf.nn.dynamic_rnn()
# TensorFlow中的RNN单元代表每个循环“单元”执行的基本操作及其相关状态的抽象概念
# 创建 rnn_cell 后，将其馈入 tf.nn.dynamic_rnn() ,该函数替换了简单实现中的 tf.scan
#并创建了一个由 rnn_cell 指定的 RNN
# =============================================================================

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

element_size = 28;time_steps = 28;num_class = 10;
batch_size = 128;hidden_layer_size = 128

_inputs = tf.placeholder(tf.float32,shape=[None,time_steps,element_size],
                         name='inputs')
y = tf.placeholder(tf.float32,shape=[None,num_class],name='inputs')

# Tensorflow built-in functions
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
outputs,_ = tf.nn.dynamic_rnn(rnn_cell,_inputs,dtype=tf.float32)

Wl = tf.Variable(tf.truncated_normal([hidden_layer_size,num_class],
                                     mean=0,stddev=.01))
bl = tf.Variable(tf.truncated_normal([num_class],mean=0,stddev=.01))

def get_linear_layer(vector):
    return tf.matmul(vector,Wl) + bl

last_rnn_output = outputs[:,-1,:]
final_output = get_linear_layer(last_rnn_output)

softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,labels=y)

cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(final_output,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction,tf.float32)))*100

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

test_data = mnist.test.images[:batch_size].\
    reshape((-1,time_steps,element_size))
test_label = mnist.test.labels[:batch_size]

for i in range(3001):
    batch_x,batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size,time_steps,element_size))
    sess.run(train_step,feed_dict={_inputs:batch_x,y:batch_y})
    
    if i % 1000 == 0:
        acc = sess.run(accuracy,feed_dict={_inputs:batch_x,y:batch_y})
        loss = sess.run(cross_entropy,feed_dict={_inputs:batch_x,y:batch_y})
        
        print("Iter "+str(i)+",Minibatch Loss= "+"{:6f}".format(loss)+
              ", Training Accuracy= "+"{:5f}".format(acc))
        
print("Testing Accuracy:",sess.run(accuracy,
                                   feed_dict={_inputs:test_data,y:test_label}))
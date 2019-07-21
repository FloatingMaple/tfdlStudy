# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

# =============================================================================
# 创建简单的文本数据，一类由奇数组成，另一类为偶数。目标是在一个有监督的文本分类任务中，
#学会将每个句子分类为奇数或偶数
# =============================================================================

batch_size = 128;embedding_dimension = 64;num_classes = 2;
hidden_layer_size = 32;times_steps = 6;element_size = 1

digit_to_word_map = {1:"One",2:"Two",3:"Three",4:"Four",5:"Five",
                     6:"Six",7:"Seven",8:"Eight",9:"Nine"}
digit_to_word_map[0] = "PAD"

even_sentences = []
odd_sentences = []
# 保留原来的句子长度，可以避免处理无用的PAD符号产生噪音。
seqlens = []

for i in range(10000):
# 为每个句子取3~6之间的随机长度（包含下界但不取上界）
    rand_seq_len = np.random.choice(range(3,7))
# 句子长度序列
    seqlens.append(rand_seq_len)
# np.random.choice(a, size=None, replace=True, p=None)
#从一维array a 或 int 数字a 中,以概率p随机选取大小为size的数据,replace表示是否重用元素,
#即抽取出来的数据是否放回原数组中，默认为true（抽取出来的数据有重复）
#range(a,b,c) 从a到b以c为步长生成一个整数列表，不包括b
    rand_odd_ints = np.random.choice(range(1,10,2),rand_seq_len) #1,3,5,7,9
    rand_even_ints = np.random.choice(range(2,10,2),rand_seq_len) #2,4,6,8
    
# 0填充（为了将所有输入放在一个张量中，需要有统一的大小，
#因此使用0或者PAD符号来填充短于6的句子） zero-padding
    if rand_seq_len<6:
        rand_odd_ints = np.append(rand_odd_ints,[0]*(6-rand_seq_len))
        rand_even_ints = np.append(rand_even_ints,[0]*(6-rand_seq_len))
        
    even_sentences.append(" ".join([digit_to_word_map[r] \
                                    for r in rand_odd_ints]))
    odd_sentences.append(" ".join([digit_to_word_map[r] \
                                  for r in rand_even_ints]))
    
# python中对于列表list [] 的操作，* 用于重复列表，+ 用于组合列表
data = even_sentences+odd_sentences
# Same seq lengths for even,odd sentences
# 此处的操作是将原来的seqlens重复一遍，因为把奇数和偶数都放进data中了
seqlens *= 2

print(even_sentences[0:6])
print(odd_sentences[0:6])
print(seqlens[0:6])


# 将单词映射到索引（单词标识符）
word2index_map = {}
index = 0
for sent in data:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

index2word_map = {index:word for word,index in word2index_map.items()}
vocabulary_size = len(index2word_map)

# 创建标签
labels = [1]*10000 + [0]*10000
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0]*2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding
    
data_indices = list(range(len(data)))
# random.shuffle 将元素进行随机排序
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]
# 将数据分成训练集和测试集
labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]
train_x = data[:10000]
train_y = labels[:10000]
train_seqlens = seqlens[:10000]
test_x = data[10000:]
test_y = labels[10000:]
test_seqlens = seqlens[10000:]

# 生成批量句子，每个句子是一个与单词对应的整数ID列表
def get_sentence_batch(batch_size,data_x,data_y,data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i].lower().split()] \
          for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x,y,seqlens

# 为数据创建占位符
_inputs = tf.placeholder(tf.int32,shape=[batch_size,times_steps])
_labels = tf.placeholder(tf.float32,shape=[batch_size,num_classes])

# seqlens for dynamic calculation
# 将示例中每个序列的长度给dynamic_rnn() ，
#tensorflow用它来停止最后一个实际序列元素之后的所有RNN步骤，超过原始长度的步骤输出为0
_seqlens = tf.placeholder(tf.int32,shape=[batch_size])
    
with tf.name_scope("embeddings"):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_dimension],
                                               -1.0,1.0),name='embedding')
    # 能够检索给定的单词索引序列中每个单词的向量
    embed = tf.nn.embedding_lookup(embeddings,_inputs)
    
# =============================================================================
# 使用tf.contrib.rnn.BasicLSTMCell() 创建一个LSTM 单元，
#将其提供给tf.nn.dynamic_rnn()
# =============================================================================
with tf.variable_scope('lstm'):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size,forget_bias=1.0)
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,embed,sequence_length=_seqlens
                                       ,dtype=tf.float32)
    
# tf.truncated_normal(shape, mean=0.0, stddev=1.0, 
#   dtype=tf.float32, seed=None, name=None)        
#从截断的正态分布中输出随机值。 shape表示生成张量的形状，mean是均值，stddev是标准差
weights = {
        'linear_layer':tf.Variable(tf.truncated_normal(
                [hidden_layer_size,num_classes],mean=0,stddev=.01))
        }
    
biases = {
        'linear_layer':tf.Variable(tf.truncated_normal(
                [num_classes],mean=0,stddev=.01))
        }

# 提取最后的输出，应用于一个线性层
final_output = tf.matmul(states[1],weights["linear_layer"]) \
     + biases["linear_layer"]

softmax = tf.nn.softmax_cross_entropy_with_logits(
        logits=final_output,labels=_labels)
    
cross_entropy = tf.reduce_mean(softmax)
    

train_step = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(_labels,1),tf.argmax(final_output,1))

accuracy = (tf.reduce_mean(tf.cast(correct_prediction,tf.float32))) * 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(1000):
        x_batch,y_batch,seqlen_batch = get_sentence_batch(batch_size,train_x,
                                                          train_y,train_seqlens)
        sess.run(train_step,feed_dict={
                _inputs:x_batch,_labels:y_batch,_seqlens:seqlen_batch})
    
        if step % 100 == 0:
            acc = sess.run(accuracy,feed_dict={
                    _inputs:x_batch,_labels:y_batch,_seqlens:seqlen_batch})
            print("Accuracy at %d: %.5f "%(step,acc))
    
    for test_batch in range(5):
        x_test,y_test,seqlen_test = get_sentence_batch(
                batch_size,test_x,test_y,test_seqlens)
        batch_pred,batch_acc = sess.run([tf.argmax(final_output,1),accuracy],
                                         feed_dict={_inputs:x_test,
                                                    _labels:y_test,
                                                    _seqlens:seqlen_test})
        print("Test batch accuracy %d: %.5f " % (test_batch,batch_acc))
        
    output_example = sess.run([outputs],feed_dict={
            _inputs:x_test,_labels:y_test,_seqlens:seqlen_test})
    states_example = sess.run([states[1]],feed_dict={
            _inputs:x_test,_labels:y_test,_seqlens:seqlen_test})
    
    
    
    
    
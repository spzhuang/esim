# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 08:22:43 2019
Notice1:
    label = 1, 表示 entailment
    label = 2, 表示 contradiction
    label = 3, 表示 neutral
Notice2:
    使用时，注意修改root路径为自己本地的语料路径
    本脚本使用了snli语料集
    导入的语料格式已经经过预处理，其格式如下：
        "[(label,premise,hypothesis),(label,premise,hypothesis),...]"即为"[(str1,str2,str3),(str1,str2,str3),...]"
        label，premise，hypothesis分别代表文本蕴含关系中的标签，假设，前提
Notice3:
    使用时，可以修改devfile与trainfile的切片数量。
Notice4：
    神经网络框架： tensorflow 1.14
Notice5:
    需要使用静态词嵌入
@author: Asus
"""


import pdb
import tensorflow as tf
import numpy as np
import time
root = 'G:\数据\snli_1.0\snli_1.0\使用python进行预处理的数据'
devfile = 'snli_dev.txt'
testfile = 'snli_test.txt'
trainfile = 'snli_train.txt'
with open(root+'\\'+devfile,'r') as f:
    devfile = eval(f.read())
with open(root+'\\'+trainfile,'r') as f:
    trainfile = eval(f.read())
devfile = trainfile[:500]    
trainfile = trainfile[500:3000]


def predeal(cor):
    '''cor type: [(label,premise,hypothesis),...]'''
    label = []
    premise = []
    hypothesis = []
    for i,j,k in cor:
        if i == 'entailment':
            label.append(1)
        if i == 'contradiction':
            label.append(2)
        if i == 'neutral':
            label.append(3)
        premise.append(j)
        hypothesis.append(k)
    return label,premise,hypothesis

Label_t,Premise_t,Hypothesis_t = predeal(trainfile)
Label_d,Premise_d,Hypothesis_d = predeal(devfile)

Cor_t = list(zip(Premise_t,Hypothesis_t))
Cor_d = list(zip(Premise_d,Hypothesis_d))
def word_info(cor):
    '''word_info: 收集词汇的信息
    输出 word2ix,word2count,ix2word,vocab_num 
    cor type: [(premise,hypothesis),...]'''
    # Type of cor is zip
    cor_a = [i+' '+j for i,j in cor]
    cor_b = [i.split() for i in cor_a]
    word2ix = {}
    ix2word = {0:'⛔'}
    word2count = {}
    vocab_num = 2
    # 定义几个字典：
    # word2ix: 从词到下标的字典
    # ix2word: 从下标到词的字典
    # word2count: 从词到词频的字典
    for i in cor_b:
        for j in i:
            if j not in word2ix:
                word2ix[j] = vocab_num
                word2count[j] = 1
                ix2word[vocab_num] = j
                vocab_num += 1
            else:
                word2count[j] += 1
    return word2ix,word2count,ix2word,vocab_num

Word2ix,Word2count,Ix2word,VocabNum = word_info(list(zip(Premise_t+Premise_d,Hypothesis_t+Hypothesis_d)))


def filiter(cor,max_seq_len):
    ''' cor type: [(premise,hypothesis),...]
    
    过滤掉cor中长度大于max_seq_len的句子对'''
    inde = []
    for i in range(len(cor)):
        len_a = len(cor[i][0].split())
        len_b = len(cor[i][1].split())
        if len_a > max_seq_len or len_b > max_seq_len:
            inde.append(i)
    for j in inde[::-1]:
        cor.pop(j)
    return cor


def line_to_array(sentence,word2ix=Word2ix,max_seq_len = 30):
    '''将sentence的内容根据word的index转化为array'''
    sen_lis = sentence.split()
    mat = np.zeros(max_seq_len)
    for i in range(len(sen_lis)):
        mat[i] = word2ix[sen_lis[i]]
    # 这里没有用到句子的开始符，这个等到实际建模再做修改，现在还不知道句子的开始符是否有作用
    return mat


def arraybatch_from_cor(cor,label,word2ix = Word2ix,max_seq_len = 30,batch_size = 100,shuffle=True):
    '''cor type: [(premise,hypothesis),...]
       label type: [label,label,...]
       word2ix type: {'word1':index1,
                      'word2':index2,
                      ...}'''
      # 先试试看把所有的batch生成一个迭代器
    if shuffle == True:
        long = len(cor)
        cor_index = list(range(long))
        np.random.shuffle(cor_index)
        cor_copy,label_copy = [],[]
        for i in range(len(cor_index)):
            cor_copy.append(cor[cor_index[i]])
            label_copy.append(label[cor_index[i]])
        cor = cor_copy
        label = label_copy
    res_number = len(cor)%batch_size
    if res_number != 0:
        trac_num = batch_size*(len(cor)//batch_size) 
        cor = cor[:trac_num]
    assert len(cor)%batch_size == 0
    batch_num = len(cor)//batch_size
#    index = 1
    batch_lis = []
    for i in range(batch_num):
        mat_premise = np.zeros(shape = [batch_size,max_seq_len],dtype = np.int32)
        mat_hypothesis = np.zeros(shape = [batch_size,max_seq_len],dtype = np.int32)
        mat_label = np.zeros(shape = [batch_size,3],dtype = np.int32)
        temp_cor = cor[i*batch_size:(i+1)*batch_size]
        label_cor = label[i*batch_size:(i+1)*batch_size]
        for vp in range(len(temp_cor)):
              premise_sentence = temp_cor[vp][0]
              mat_pre = line_to_array(premise_sentence,max_seq_len = max_seq_len)
              mat_premise[vp] = mat_pre
              hypothesis_sentence = temp_cor[vp][1]
              mat_hyp = line_to_array(hypothesis_sentence,max_seq_len = max_seq_len)
              mat_hypothesis[vp] = mat_hyp
              mat_label[vp,label_cor[vp]-1] = 1
        batch_lis.append([mat_label,mat_premise,mat_hypothesis])
    return batch_lis,len(batch_lis)

##  要把Batch里面dtype调整为tf.int32
class ESIM():
    # 终于，到了最不确定的一环啦 😄 😄 😄
    # 当使用tensorflow时，如何建立模型，不是一个简单的问题，for me
    def __init__(self,vocab_num:int,embedding_dim:int,batch_size:int,
                 max_seq_len:int,lstm_hidden_dim:int,dropout:float):
        # 初始化的逻辑是定义必要的变量
        # input.shape = [100,30]
        self.p_input = tf.placeholder(tf.int32,[batch_size,max_seq_len])
        self.h_input = tf.placeholder(tf.int32,[batch_size,max_seq_len])
        self.real_label = tf.placeholder(tf.int32,[batch_size,3])
        self.dropout = tf.placeholder(tf.float32)
        
        self.lstm_hidden_dim = lstm_hidden_dim
        self.max_seq_len = max_seq_len
        
        
        with tf.variable_scope('embedding',reuse = tf.AUTO_REUSE):
            self.embedding_w = tf.Variable(initial_value = tf.random.normal([vocab_num,embedding_dim],0,0.1))
            self.p_embedding = self.droptensor(tf.nn.embedding_lookup(self.embedding_w,self.p_input),self.dropout)
            self.q_embedding = self.droptensor(tf.nn.embedding_lookup(self.embedding_w,self.h_input),self.dropout)
            
        # step1 Input by BiLSTM
        with tf.variable_scope('input_bilstm',reuse = tf.AUTO_REUSE):
            lstm_cell_fw1 = tf.nn.rnn_cell.LSTMCell(num_units = 300,dtype = tf.float32)
            lstm_cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell = lstm_cell_fw1, input_keep_prob = 1-0.2) # 1-hparam.dropout
            lstm_cell_bw1 = tf.nn.rnn_cell.LSTMCell(num_units = 300,dtype = tf.float32)            
            lstm_cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell = lstm_cell_bw1, input_keep_prob = 1-0.2)
            pre_out,pre_hidden= tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_cell_fw1,
                                    cell_bw = lstm_cell_bw1, inputs = self.p_embedding,dtype = tf.float32)
            a1 = tf.concat([pre_out[0],pre_out[1]],axis = 2)
            
            hyp_out,hyp_hidden = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_cell_fw1,
                                    cell_bw = lstm_cell_bw1, inputs = self.q_embedding,dtype = tf.float32)            
            b1 = tf.concat([hyp_out[0],hyp_out[1]],axis = 2)
            
#            self.a1_hidden = tf.concat((self.pre_hidden[0][1],self.pre_hidden[1][1]),axis = -1)
#            self.b1_hidden = tf.concat((self.hyp_hidden[0][1],self.hyp_hidden[1][1]),axis = -1)
            
        # step2 Local Inference Modeling
        with tf.variable_scope('local_infer'):
            infer_pre = tf.split(a1,batch_size,0)
            infer_hyp = tf.split(b1,batch_size,0)
            ## self.infer_pre : [tensor,tensor,...]
            ##                  [shape(1,max_time,embedding_dim)] 
            pre_batch = len(infer_pre)
            attention_batch = [0 for i in range(pre_batch)]
            for i in range(pre_batch):
                pre_sentence = infer_pre[i]
                pre_sentence = tf.squeeze(pre_sentence, axis = 0)
                # pre_sentence.shape = [max_seq_len,lstm_out_dim] (30,400)
#                pre_shape = pre_sentence.shape.as_list()
#                pre_word = tf.split(pre_sentence,pre_shape[0],0)
                
                hyp_sentence = infer_hyp[i]
                hyp_sentence = tf.squeeze(hyp_sentence, axis = 0)
                # pre_sentence.shape = [max_seq_len,lstm_out_dim] (30,400)

#                hyp_shape = hyp_sentence.shape.as_list()
#                hyp_word = tf.split(hyp_sentence,hyp_shape[0],0)
                # =====计算attention矩阵的值
                attention_ph = tf.matmul(pre_sentence,tf.transpose(hyp_sentence))
                sum_attn = tf.reduce_mean(attention_ph,axis=1)
                sum_attn = tf.expand_dims(sum_attn,1)
                sum_attn = tf.tile(sum_attn,[1,attention_ph.shape.as_list()[1]])
                attention_ph = tf.math.divide(attention_ph,sum_attn)

                attention_hp = tf.matmul(hyp_sentence,tf.transpose(pre_sentence))
                sum_attn = tf.reduce_mean(attention_hp,axis=1)
                sum_attn = tf.expand_dims(sum_attn,1)
                sum_attn = tf.tile(sum_attn,[1,attention_hp.shape.as_list()[1]])
                attention_hp = tf.math.divide(attention_hp,sum_attn)
                attention_hp = tf.stack(attention_hp)
                
                attention_batch[i] = (attention_ph,attention_hp)
            # 至此，我们得到了每个batch的一个存储attention的列表(且指数归一化了)
            # 且列表的元素就是batch中每个对应的p和h两句话构成的attention矩阵
            # 该矩阵的i,j元素表示p的第i个词和q的第j个词的embedding的内积
            
            # 现在计算a2，b2
            a2_batch = []  # 该列表用来存储该批次每个premise样本产生的a2，b2_batch类似
            b2_batch = []
            for i in range(pre_batch):
                matrix_ph = attention_batch[i][0]
                matrix_hp = attention_batch[i][1]
                
                tem_b =  infer_hyp[i]
                temp_a = tf.matmul(matrix_ph,tem_b)
                sen_a2 = tf.reduce_sum(temp_a,axis = 0)
                
                tem_a =  infer_pre[i]
                temp_b = tf.matmul(matrix_hp,tem_a)
                sen_b2 = tf.reduce_sum(temp_b,axis = 0)
                
                a2_batch.append(sen_a2)
                b2_batch.append(sen_b2)
            
            # ========至此，我们已经计算完成a2,b2，接下来连接[a1,a2,a1-a2,a1⊙a2]
            a2 = tf.stack(a2_batch)
            b2 = tf.stack(b2_batch)
            ## self.a2.shape = [batch_size,max_time,embedding_dim]
            ma = tf.concat([a1,a2,tf.abs(a1-a2),tf.math.multiply(a1,a2)],axis = -1)
            mb = tf.concat([b1,b2,tf.abs(b1-b2),tf.math.multiply(b1,b2)],axis = -1)
            
        #==== step3: Inference Composition
        with tf.variable_scope('Inference',reuse=tf.AUTO_REUSE):
            shape_a = ma.shape.as_list()[2]
            shape_b = mb.shape.as_list()[2]

            # imap 是用来降维的
            
            w3_a = tf.Variable(tf.random.normal([shape_a,self.lstm_hidden_dim],0,1),dtype = tf.float32)
            w3_b = tf.Variable(tf.random.normal([shape_b,self.lstm_hidden_dim],0,1),dtype = tf.float32)
            ma_f = tf.nn.relu(tf.matmul(ma,w3_a))
            mb_f = tf.nn.relu(tf.matmul(mb,w3_b))
            
            lstm_cell_fw2 = tf.nn.rnn_cell.LSTMCell(num_units = 300,dtype = tf.float32)
            lstm_cell_fw2 = tf.contrib.rnn.DropoutWrapper(cell = lstm_cell_fw2, input_keep_prob = 1-0.2) # 1-hparam.dropout
            lstm_cell_bw2 = tf.nn.rnn_cell.LSTMCell(num_units = 300,dtype = tf.float32)            
            lstm_cell_bw2 = tf.contrib.rnn.DropoutWrapper(cell = lstm_cell_bw2, input_keep_prob = 1-0.2)
            
            
            # 前后两个的rnn和的神经元要一致，否则就不能够使用前面输出的状态层了
            va_out,_ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw2,lstm_cell_bw2,
                            ma_f,dtype = tf.float32)
            vb_out,_ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw2,lstm_cell_bw2,
                            mb_f,dtype=tf.float32)
            
        with tf.variable_scope('full_connect',reuse=tf.AUTO_REUSE):

            va = tf.concat([va_out[0],va_out[1]],axis=-1)
            vb = tf.concat([vb_out[0],vb_out[1]],axis=-1)
            va_max = tf.reduce_max(va,axis = 1)
            vb_max = tf.reduce_max(vb,axis = 1)
            va_mean = tf.reduce_mean(va, axis = 1)
            vb_mean = tf.reduce_mean(vb, axis = 1)
            self.v = tf.concat([va_max,va_mean,vb_max,vb_mean],axis = -1)
            v_long = self.v.shape.as_list()[-1]
            w4 = tf.Variable(tf.random.normal([v_long,self.lstm_hidden_dim],0,1),dtype = tf.float32)
            b4 = tf.Variable(tf.random.normal([self.lstm_hidden_dim],0,1),dtype = tf.float32)
            mid  = tf.nn.relu(tf.matmul(self.v,w4)+b4)
            
            w5 = tf.Variable(tf.random.normal([self.lstm_hidden_dim,3],0,1),dtype = tf.float32)
            b5 = tf.Variable(tf.random.normal([3],0,1),dtype = tf.float32)
            self.logits = tf.matmul(mid,w5)+b5
            
            
        # =================构建损失函数和优化器
        with tf.variable_scope('loss_optimizer',reuse=tf.AUTO_REUSE):
            lossfun = tf.nn.softmax_cross_entropy_with_logits
            loss = lossfun(labels = self.real_label,logits = self.logits)
            self.loss = tf.reduce_mean(loss)
            equal = tf.equal(tf.argmax(self.real_label,1),tf.argmax(self.logits,1))
            self.acc = tf.reduce_sum(tf.cast(equal,tf.float32))
            optimizer = tf.train.AdamOptimizer(0.001,beta1=0.9, beta2=0.999,epsilon=1e-8)
            self.opdate = optimizer.minimize(self.loss)
    def droptensor(self,tensor,keeprob):
        return tf.nn.dropout(tensor,keeprob)
# 参数区     
EmbeddingDim = 300  
LstmHiddenDim = 300
BatchSize = 32
MaxSeqLen = 50
Dropout = 0.5     
model = ESIM(vocab_num=VocabNum,embedding_dim=EmbeddingDim,batch_size=BatchSize,
                 max_seq_len=MaxSeqLen,lstm_hidden_dim=LstmHiddenDim,dropout=Dropout)
#model_dev = ESIM(infer = True,vocab_num=Vocab_num,embedding_dim=Embedding_dim,batch_size=Batch_size,max_seq_len=30)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print('批次大小：%d'%BatchSize)
Epochs = 5

res = [[] for i in range(Epochs)]
Cor_t = filiter(cor = Cor_t, max_seq_len = MaxSeqLen)
Cor_d = filiter(cor = Cor_d, max_seq_len = MaxSeqLen)
Batch,nt = arraybatch_from_cor(Cor_t,Label_t,batch_size=BatchSize,max_seq_len = MaxSeqLen,shuffle=True)
Batch_d,nd = arraybatch_from_cor(Cor_d,Label_d,batch_size=BatchSize,max_seq_len = MaxSeqLen)
Sample_sum_tr = nt*BatchSize
Sample_sum_dv = nd*BatchSize

#ppd = []
start = time.time()
for epoch in range(Epochs):
    print('第%d次训练'%(epoch+1))
    ##  由于双向lstm有两个方向，因此pre的状态是由嵌套元组构成的。
    tr_loss,tr_acc = 0,0 
    te_loss,te_acc = 0,0
    for t,[i,j,k] in enumerate(Batch):
        Feed_dict = {model.p_input:j, model.h_input:k, model.real_label:i, model.dropout:Dropout}
        loss_value,acc,_ = sess.run([model.loss,model.acc,model.opdate],feed_dict = Feed_dict)
        tr_loss += loss_value
        tr_acc += acc
        if t % 4 == 0:
            print('迭代次数：%2d,损失值：%.3f, 精确率：%.1f%%'%(t,loss_value,acc/BatchSize*100))
    tr_loss,tr_acc = tr_loss /t,tr_acc / Sample_sum_tr    
    for t,[i,j,k] in enumerate(Batch_d):
        dev_dict = {model.p_input:j, model.h_input:k, model.real_label:i, model.dropout:Dropout}
        loss_value2,acc2 = sess.run([model.loss,model.acc],feed_dict = dev_dict)
        te_loss += loss_value2
        te_acc += acc2
    te_loss,te_acc = te_loss /t,te_acc/Sample_sum_dv
    timeuse = time.time() - start
    minutes = int(timeuse // 60)
    seconds = int(timeuse % 60)
    print('epoch = %2d'%(epoch+1))
    print('时间已用: %2dmin,%2dsec'%(minutes,seconds))
    print('训练集, train loss = %.4f, train accuracy = %.1f%%'%(tr_loss,tr_acc*100))
    print('测试集, dev loss = %.4f, dev accuracy = %.1f%%'%(te_loss,te_acc*100))
    
           

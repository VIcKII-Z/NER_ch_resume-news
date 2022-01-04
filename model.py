import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from utils import get_logger
from eval import conlleval


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):

        # 批次大小
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF  # True
        self.update_embedding = args.update_embedding
        # drop操作参数
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer  # Adam
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        # tag2label = {"O": 0,
        # "B-PER": 1, "I-PER": 2,
        # "B-LOC": 3, "I-LOC": 4,
        # "B-ORG": 5, "I-ORG": 6
        # }
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config

    def build_graph(self):
        # 占位符
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        # 损失函数
        self.loss_op()
        self.trainstep_op()
        # 初始化所有变量
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids')  # 原本的字的id列表，以句为单位：二维
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')  # 原本的标签序列
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name='sequence_lengths')  # 一个样本的原本的序列长度列表，一维，即句子们的长度列表
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')  # dropout
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def lookup_layer_op(self):
        with tf.variable_scope('words'):
            # 设定字嵌入变量
            _word_embeddings = tf.Variable(self.embeddings,  # 默认调用data.py中的正态分布随机数，vocablen*300维的矩阵
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           # 默认是True，如果为True，则会默认将变量添加到图形集合GraphKeys.TRAINABLE_VARIABLES中。此集合用于优化器Optimizer类优化的的默认变量列表【可为optimizer指定其他的变量集合】，可就是要训练的变量列表。这样的话在训练的过程中就会改变值
                                           name='_word_embeddings')
            # tf.nn.embeddings_lookup(),返回param张量里面索引对应的元素
            word_embeddings = tf.nn.embeddings_lookup(params=_word_embeddings,
                                                      ids=self.word_ids,
                                                      name='word_embeddings')
            # tf.nn.dropout()函数
            self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

            print('model 93行：')
            print(self.word_embeddings.shape)
            print('model 97:')
            print(self.word_ids)

    def biLSTM_layer_op(self):
        with tf.variable_scope('bi-lstm'):
            cell_fw = LSTMCell(self.hidden_dim)  # 隐藏层神经元维数，这里embedding为300，即为300维
            cell_bw = LSTMCell(self.hidden_dim)
        # tf.nn.bidirectional_dynamic_rnn()函数
        # 第一个返回outputs:(output_fw_seq,output_bw_seq),即是一个包含前向cell输出tensor和后向cell输出tensor组成的二元组。tensor的形状默认为[batch_size, max_time, depth]，可调time_major参数为True改成[max_time, batch_size, depth]。

        # 第二个返回以LSTMStateTuple型返回output_states,即包含了前向和后向最后的隐藏状态的组成的二元组,LSTMStateTuple由（c,h）组成，即c，h矩阵，memory cell state和hidden state
        (output_fw_seq, output_bw_seq), __ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=self.word_embeddings,
            sequence_lengths=self.sequence_lengths,
            dtype=tf.float32
        )
        print('model 137')


        print(self.word_embeddings)
        print('model 143')
        print(output_fw_seq.shape)
        print(output_bw_seq.shape)

        # 在最后一维进行拼接，即两个depth为300的现在拼成[batch_size, max_time, 600]
        output = tf.concat([output_fw_seq, output_fw_seq], axis=-1)
        print('model 151')
        print(output.shape)

        with tf.variable_scope('proj'):
    # 设定weight矩阵
            W = tf.get_variable(name='W',# 与tf.Variable不一样，get_variable必须命名，之后用名字调变量，所以同一个scope不能重复命名除了调参数reuse=True
                        shape = [2 * self.hidden_dim, self.num_tags],  # [600,7]
            # tf.contrib.layers.xavier_initializer()函数
            # 返回一个用于初始化权重的初始化程序 “Xavier” 。
            # 这个初始化器是用来保持每一层的梯度大小都差不多相同
                        initializer = tf.contrib.layers.xavier_initializer(),
                        dtype = tf.float32)

    # 设定bias
            b = tf.get_variable(name='b',
                        shape=[self.num_tags],  # [7]
                        initializer=tf.zeros_initializer(),  # 初始化函数tf.zeros_initializer()，也可以简写为tf.Zeros()
                        dtype=tf.float32
                        )

    # 之前output形状为[batch_size,steps,cell_num]，为了与W做乘法，做reshape处理
            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])  # reshape()函数可自动将原数组改成想要的形状，可以允许有一个-1来代替维度，会自动算出那个维度应该是多少
            pred = tf.malmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])  # [-1,batch_size,7]

            print("******************************************************************************")
            print(self.logits.shape)


    def loss_op(self):
    # 如果需要crf层
        if self.CRF:
        # crf_log_likelihood()函数，返回损失函数和转移概率
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.labels,
                                                                    sequence_lengths=self.sequence_lengths)
        # reduce_mean()指定维度的平均值
            self.loss = -tf.reduce_mean(log_likelihood)
    # 如果不要crf层
        else:
    # 交叉熵做损失函数
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logit=self.logits, labels=self.labels)
    # sequence_mask()函数，布尔类型张量作为滤波器掩模，过滤掉我们不想要的的值
            mask = tf.sequence_mask(self.sequence_lengths)
    # boolean_mask(a,b)函数，只保留a中与b中True值同下标的元素
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)


        tf.summary.scalar('loss', self.loss)


    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)  # 取最大索引
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)


    def trainstep_op(self):
        with tf.variable_scope('train_step'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            # 多种优化器选择
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            # 计算梯度
            grads_and_vars = optim.compute_gradients(self.loss)
            # 使用计算得到的梯度来更新对应的variable
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]  # tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    ##train和test过程，从这里开始写的是溯源形式，涉及相关函数在之后定义
    def train(self, train, dev):
        """
        :param train:train_data,一句为一个tuple，tuple第一个元素为单字组成的列表，第二个元素为对应标签
        :param dev:test_data
        """
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self._init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('============testing============')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    # 用模型测试句子
    def demo_one(self, sess, sent):
        label_list = []
        for seqs, labels, in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            # 例子：小明的大学在北京的北京大学
            # seqs：[[841, 37, 8, 55, 485, 73, 87, 74, 8, 87, 74, 55, 485]]
            # labels：[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            label_list_, _ = self.predict_one_batch(sess, seqs)
            # 预测值：label_list：[[1, 2, 0, 0, 0, 0, 3, 4, 0, 5, 6, 6, 6]]
            label_list.extend(label_list_)
        # 将预测出的数字label又转换成可读的原tag
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        # label2tag:{0: 0, 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC', 5: 'B-ORG', 6: 'I-ORG'}
        tag = [label2tag[label] for label in label_list[0]]  # 就一个句子，取的list第一个元素，可改。
        # tag: ['B-PER', 'I-PER', 0, 0, 0, 0, 'B-LOC', 'I-LOC', 0, 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG']

        return tag

    # 一次迭代的训练
    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):

        # 计算出batch数
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        # 记录开始训练的时间
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 产生每一个batch
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):  # enumarate 同时列出元素下标和元素
            sys.stdout.write('processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            # 此时已经执行了的step数
            step_num = epoch * num_batches + step + 1
            #
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _,loss_train,summary,step_num_=sess.run([self.train_op,self.loss,self.merged,self.global_step],feed_dict=feed_dict)
            #每隔300step记录一次
            if step + 1 == 1 or (step+1)%300 == 0 or step+1 == num_batches:
                self.logger.info('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))
            #可视化 tensorboard
            self.file_writer.add_summary(summary,step_num)

            #到最后一个batch跑完后保存模型
            if step+1 ==num_batches:
                saver.save(sess,self.model_path,global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev,seq_len_list_dev= self.dev_one_epoch(sess,dev)
        self.evaluate(label_list_dev,seq_len_list_dev,dev,epoch)

##以下为关于trainr test的一些函数
    def get_feed_dict(self,seqs,labels=None,lr=None,dropout=None):

        word_ids,seq_len_list=pad_sequences(seqs,pad_mark=0)#word_ids:用0补齐后句子的序列的列表，seq_len_list:句子的真实长度列表
        feed_dict={self.word_ids:word_ids,self.sequence_length:seq_len_list}
        if labels is not None:
            #label 也要进行补齐再喂给dict
            labels_,_=pad_sequences(labels,pad_mark=0)
            feed_dict[self.labels]=labels_
        if lr is not None:
            feed_dict[self.lr_pl]=lr
        if dropout is not None:
            feed_dict[self.dropout_pl]=dropout

        return feed_dict,seq_len_list

    #跑一次验证集
    def dev_one_epoch(self,sess,dev):
        label_list,seq_len_list=[],[]
        for seqs,labels in batch_yield(dev,self.batch_size,self.vocab,self.tag2label,shuffle=False):
            label_list_,seq_len_list_=self.predict_one_batch(sess,seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list)
        return label_list,seq_len_list

    #predict一个batch
    def predict_one_batch(self,sess,seqs):
        feed_dict,seq_len_list=self.get_feed_dict(seqs,dropout=1.0)
        if self.CRF:
            logits,transition_params =sess.run([self.logits,self.transition_params],feed_dict=feed_dict)
            print(logits.shape)  # 例子里为1*13*7
            print(transition_params)  # 7*7矩阵
            label_list = []
            print(seq_len_list)#[13]
            for logit,seq_len in zip(logits,seq_len_list):
                #viterbi_decode函数,将logits解析得到一个数
                viterbi_seq,_=viterbi_decode(logit[:seq_len],transition_params)
                label_list.append(viterbi_seq)
            print('*-*******************************************************')
            print(label_list) # 对logit按行解析返回的值[[1, 2, 0, 0, 0, 0, 3, 4, 0, 5, 6, 6, 6]]#这就是预测结果，对应着tag2label里的值
            return label_list.seq_len_list
        #不用crf
        else:
            #取logit值最大的
            label_list=sess.run(self.labels_softmax_,feed_dict=feed_dict)
            return label_list,seq_len_list


    def evaluate(self,label_list,seq_len_list,data,epoch=None):
        label2tag={}
        for tag,label in self.tag2label.items():
            label2tag[label]=tag if label !=0 else label

        model_predict =[]
        for label_,(sent,tag) in zip(label_list,data):
            tag_=[label2tag[label__] for label__ in label_]
            sent_res=[]
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i],tag[i],tag_[i]])
            model_predict.append(sent_res)

            #保存结果文件
            epoch_num=str(epoch+1) if epoch!=None else 'test'
            label_path=os.path.join(self.result_path,'label_'+epoch_num)
            metric_path=os.path.join(self.result_path,'rusult_metric_'+epoch_num)

            #使用conlleval.pl对cef测试结果进行评价的方法
            for _ in conlleval(model_predict,label_path,metric_path):
                self.logger.info(_)









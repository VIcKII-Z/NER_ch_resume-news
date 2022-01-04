import os
import pickle
import random
import codecs
import numpy as np

# 三分类标签BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }

# 默认数据集 MSRA tags, BIO
tag2label_msra = {"O": 0,
                  "B-PER": 1, "I-PER": 2,
                  "B-LOC": 3, "I-LOC": 4,
                  "B-ORG": 5, "I-ORG": 6
                  }

# 人民日报数据集
tag2label_chinadaily = {"O": 0,
                        "B-PERSON": 1, "I-PERSON": 2,
                        "B-LOC": 3, "I-LOC": 4,
                        "B-ORG": 5, "I-ORG": 6,
                        "B-GPE": 7, "I-GPE": 8,
                        "B-MISC": 9, "I-MISC": 10
                        }
# WeiboNER
tag2label_weibo_ner = {"O": 0,
                       "B-PER.NAM": 1, "I-PER.NAM": 2,
                       "B-LOC.NAM": 3, "I-LOC.NAM": 4,
                       "B-ORG.NAM": 5, "I-ORG.NAM": 6,
                       "B-GPE.NAM": 7, "I-GPE.NAM": 8,
                       "B-PER.NOM": 9, "I-PER.NOM": 10,
                       "B-LOC.NOM": 11, "I-LOC.NOM": 12,
                       "B-ORG.NOM": 13, "I-ORG.NOM": 14
                       }

# Resume_NER
tag2label_resume_ner = {"O": 0,
                        "B-NAME": 1, "M-NAME": 2, "E-NAME": 3, "S-NAME": 4,
                        "B-RACE": 5, "M-RACE": 6, "E-RACE": 7, "S-RACE": 8,
                        "B-CONT": 9, "M-CONT": 10, "E-CONT": 11, "S-CONT": 12,
                        "B-LOC": 13, "M-LOC": 14, "E-LOC": 15, "S-LOC": 16,
                        "B-PRO": 17, "M-PRO": 18, "E-PRO": 19, "S-PRO": 20,
                        "B-EDU": 21, "M-EDU": 22, "E-EDU": 23, "S-EDU": 24,
                        "B-TITLE": 25, "M-TITLE": 26, "E-TITLE": 27, "S-TITLE": 28,
                        "B-ORG": 29, "M-ORG": 30, "E-ORG": 32, "S-ORG": 33,
                        }

tag2label_mapping = {
    'MSRA': tag2label_msra,
    '人民日报': tag2label_chinadaily,
    'WeiboNER': tag2label_weibo_ner,
    'ResumeNER': tag2label_resume_ner

}


# 将预训练模型中的embedding字向量规范成本模型的格式，注意pre中没有的字还是用word2id里的embedding方法
def build_character_embeddings(pretrained_emb_path, embeddings_path, word2id, embedding_dim):
    print('loading pretrained embeddings from {}'.format(pretrained_emb_path))
    pre_emb = {}
    for line in codecs.open(pretrained_emb_path, 'r', 'utf-8'):
        line = line.strip().split()
        if len(line) == embedding_dim + 1:
            pre_emb[line[0]] = [float(x) for x in line[1:]]  # 预训练文件以[字,x,x,x,.....,x]的列表形式存在,转化为{字:[x,x,x,....],...}形式}
    word_ids = sorted(word2id.items(), key=lambda x: x[1])  # word2id按id升序排列
    characers = [c[0] for c in word_ids]
    embeddings = list()
    for i, ch in enumerate(characers):
        if ch in pre_emb:
            embeddings.append(pre_emb[ch])
        else:
            embeddings.append(np.random.uniform(-0.25, 0.25, embedding_dim).tolist())
        embeddings = np.asarray(embeddings, dtype=np.float32)
        np.save(embeddings_path, embeddings)


##读取数据集
def read_corpus(corpus_path):
    """
    :param corpus_path数据集路径
    :return: data()
    """
    data = []
    # 打开文件
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()

    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()  # data格式： #寇    O
            # 在	O
            # 京	B-LOC
            sent_.append(char)
            tag_.append(label)
        ##以换行符为分隔
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
        return data


##词典的创建{字：id（数字）} embedding
def vocab_build(vocab_path, corpus_path, min_count):
    """
    :param vocab_path
    :param mincount:低频词阈值
    :return none 写了Word2id词典

    """
    data = read_corpus(corpus_path)  # 调用刚刚的读取数据集函数
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit(): # 字本身是数字的情况
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):  # 字本身是英文字母的情况
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]  # 字典格式[顺序，次数]：{'当': [1, 2], '希': [2, 1], '望': [3, 1]}
            else:
                word2id[word][1] += 1
        low_freq_words = []  # 列出并删除低频（小于阈值min_count）出现的词
        for word, [word_id, word_freq] in word2id.items():
            if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
                low_freq_words.append(word)
        for word in low_freq_words:
            del word2id[word]

        new_id = 1
        for word in word2id.keys():
            word2id[word] = new_id
            new_id += 1
        word2id['<UNK>'] = new_id
        word2id['<PAD>'] = 0

        print(len(word2id))
        with open(vocab_path, 'wb') as fw:
            pickle.dump(word2id, fw)  # pickle 模块 序列化对象

    ##按照刚刚生成的Word2id词典生成句子id
def sentence2id(sent, word2id):
    """
    :param sent
    :param word2id
    :return sentence_id 字的列表形式的句子id

    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

    ##读取词典
    def read_dictionary(vocab_path):
        """
        :param 词典相对路径

        """
        vocab_path = os.path.join(vocab_path)
        with open(vocab_path, 'rb') as fr:
            word2id = pickle.load(fr)  # 反序列化,将词典load过来
        print('vocab_size', len(word2id))
        return word2id

    def random_embedding(vocab, embedding_dim):
        """
        :param
        :return
        """
        embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
        # 正态分布的而随机数作为embedding,返回lenvocab*embedingdim的矩阵，即每个word in vocab 用embeddingdim维向量表示，作为初始值
        # numpy.random.uniform(low,high,size)功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.

        embedding_mat = np.random.float(embedding_mat)

    # 对句子进行补齐，默认为用0,使句子长度都一样
    def pad_sequences(sequences, pad_mark=0):
        """
        :return 补齐后的句子列表以及句子之前的真实长度列表
        """
        max_len = max(map(lambda x: len(x), sequences))  # 取最大的sequence长度
        # map()映射
        # >>>def square(x) :            # 计算平方数
        # ...     return x ** 2
        # ...
        # >>> map(square, [1,2,3,4,5])   # 计算列表各个元素的平方
        # [1, 4, 9, 16, 25]
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)  # 由元组格式转化为列表格式
            seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))  # 记录补齐前的真实长度
        return seq_list, seq_len_list


# 生成batch
def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    if shuffle:
        random.shuffle(data)  # 把句子打乱，默认为false
    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)  # 用vocab（词频）来id化句子里的每个单字，形成句子
        # sent_的形状为[33,12,17,88,50....]
        label_ = [tag2label[tag] for tag in tag_]  # 换成对应数字
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []
        seqs.append(sent_)
        labels.append(label_)
    if len(seqs) != 0:
        yield seqs, labels








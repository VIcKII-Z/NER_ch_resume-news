import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label_mapping, random_embedding, vocab_build, \
    build_character_embeddings

# Session configuration
#设置GPU信息
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
#tf.ConfigProto(log_device_placement=True)#记录设备指派情况，可以获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,会在终端打印出各项操作是在哪个设备上运行的。
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # need ~700MB GPU memory

# hyper parameters超参数设置
# 使用argparse的第一步就是创建一个解析器对象，并告诉它将会有些什么参数。 那么当程序运行时，该解析器就可以用于处理命令行参数
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--dataset_name', type=str, default='MSRA',
                    help='choose a dataset(MSRA, ResumeNER, WeiboNER,人民日报)')
# parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
# parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=20, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--use_pre_emb', type=str2bool, default=False,
                    help='use pre_trained char embedding or init it randomly')
parser.add_argument('--pretrained_emb_path', type=str, default='sgns.wiki.char', help='pretrained embedding path')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()

# vocabulary build
if not os.path.exists(os.path.join('data_path', args.dataset_name, 'word2id.pkl')):
    vocab_build(os.path.join('data_path', args.dataset_name, 'word2id.pkl'),
                os.path.join('data_path', args.dataset_name, 'train_data.txt'))

# get word dictionary
word2id = read_dictionary(os.path.join('data_path', args.dataset_name, 'word2id.pkl'))

# build char embeddings
if not args.use_pre_emb:
    embeddings = random_embedding(word2id, args.embedding_dim)
    log_pre = 'not_use_pretrained_embeddings'
else:
    pre_emb_path = os.path.join('.', args.pretrained_emb_path)
    embeddings_path = os.path.join('data_path', args.dataset_name, 'pretrain_embedding.npy')
    if not os.path.exists(embeddings_path):
        build_character_embeddings(pre_emb_path, embeddings_path, word2id, args.embedding_dim)
    embeddings = np.array(np.load(embeddings_path), dtype='float32')
    log_pre = 'use_pretrained_embeddings'

# choose tag2label#选数据集
tag2label = tag2label_mapping[args.dataset_name]

# read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('data_path', args.dataset_name, 'train_data.txt')#默认datapath/MSRA/train_data.txt
    test_path = os.path.join('data_path', args.dataset_name, 'test_data.txt')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    test_size = len(test_data)

# paths setting 设置保存文件路径，各种文件path用字典保存
paths = {}
#训练模式用时间戳命名
# 时间戳就是一个时间点，一般就是为了在同步更新的情况下提高效率之用。
# 就比如一个文件，如果他没有被更改，那么他的时间戳就不会改变，那么就没有必要写回，以提高效率，
# 如果不论有没有被更改都重新写回的话，很显然效率会有所下降。
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('model_path', args.dataset_name, timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, args.dataset_name + log_pre + "_log.txt")
paths['log_path'] = log_path
#调用utils里的get_logger函数调出log，再写入info
get_logger(log_path).info(str(args))

# training model
if args.mode == 'train':

    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    #创建节点,无返回值
    model.build_graph()

    # hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    # train model on the whole training data
    print("train data: {}".format(len(train_data)))
    print("test data: {}".format(test_size))
    model.train(train=train_data, dev=test_data)  # use test_data.txt as the dev_data to see overfitting phenomena

# testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

# demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while (1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                PER, LOC, ORG = get_entity(tag, demo_sent)
                print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
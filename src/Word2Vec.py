"""
word2vec 的skip-gram模型，负采样的实现
"""
import torch as t
import random
import datetime
import json
from torch import optim
from data_processing import create_dict
from model import model


class Word2Vec(object):
    config = json.load(open('../config/config.json', 'r', encoding='utf8'))

    def __init__(self,
                 inputfile,
                 outputfile,
                 word_emb_dim=config["word_dim"],
                 lr=config['lr'],
                 train_times=config['train_times'],
                 batch_size=config['batch_size'],
                 window_size=config['window_size'],
                 min_count=config['min_count']):
        # skip-gram模型中的窗口的大小
        self.window_size = window_size
        self.inputfile = inputfile
        self.word2id, self.id2word = create_dict(self.inputfile, min_count)
        # 中心词和上下文列表
        self.center_context_list = self.get_center_w()
        # print(len(self.center_context_list))
        # 词向量的个数
        self.emb_size = len(self.word2id)
        # 词向量的维度
        self.word_dim = word_emb_dim
        # 每批传入的词的个数
        self.batch_size = batch_size
        # 学习率
        self.lr = lr
        # 结果向量的保存的文件
        self.outputfile = outputfile
        # 模型
        self.skip_gram = model(self.emb_size, self.word_dim)
        # 是否使用cuda
        self.use_cuda = t.cuda.is_available()
        # 遍历完整个正例所需要的次数
        self.num_batch = len(self.center_context_list) // self.batch_size + 1
        if self.use_cuda:
            self.skip_gram.cuda()
        # 优化器
        self.optimizer = optim.SGD(self.skip_gram.parameters(), lr=self.lr)
        # 训练次数
        self.train_times = train_times

    def get_pos_pairs(self, idx):
        """
        每次获取语料中的正例
        :param idx: 批次的索引
        :return: 中心词列表，上下文词列表
        """
        # 语料中所有的中心词与上下文组成的对
        index_tuple = self.center_context_list[self.batch_size * idx:self.batch_size * (idx + 1)]
        # 中心词组成的列表
        pos_u = []
        # 中心词上下文组成的列表
        pos_v = []
        for center_word, context_word in index_tuple:
            pos_u.append(center_word)
            pos_v.append(context_word)
        return pos_u, pos_v, index_tuple

    def get_center_w(self):
        """
        得到每个中心词在windows_size下的上下文元组列表，去除出现频率少于一定范围的字
        :return: 中心词与上下文list
        """
        f = open(self.inputfile, 'r', encoding='utf8')
        # 存储中心词和上下文的列表
        center_context_list = []
        for line in f:
            line = line.strip().split(' ')
            # print(length)
            id_list = [self.word2id[i] for i in line if i in self.word2id]
            length = len(id_list)
            for index_i, center in enumerate(id_list):
                for j in range(self.window_size):
                    if index_i - j > 0:
                        center_context_list.append((center, id_list[index_i - j - 1]))
                    if index_i + j + 1 < length:
                        center_context_list.append((center, id_list[index_i + j + 1]))
            # print(len(center_context_list))
            # print(center_context_list)zhihu1.txt
        return center_context_list

    def neg_sample(self, pos_u, index_tuple):
        """
        负采样
        :param pos_u:正例中心词组成的列表
        :param index_tuple: 这个批次中传入的中心词与上下文列表
        :return: 负例上下文词组成的列表
        """
        # 负采样的上下文列表
        neg_v = []
        for i in range(len(pos_u)):
            pos_word_id = pos_u[i]
            while True:
                # 采样负例
                e = random.randint(0, self.emb_size - 1)
                if (pos_word_id, e) in index_tuple:
                    continue
                else:
                    neg_v.append(e)
                    break
        return neg_v

    def train(self):
        print("training time is: ", self.train_times)
        print("word count is: ", len(self.word2id))
        for epoch in range(self.train_times):
            process_bar = range(self.num_batch)
            print('train step is: ', self.num_batch)
            for i in process_bar:
                pos_u, pos_v, index_tuple = self.get_pos_pairs(i)
                neg_v = self.neg_sample(pos_u, index_tuple)
                if self.use_cuda:
                    pos_u = t.LongTensor(pos_u).cuda()
                    pos_v = t.LongTensor(pos_v).cuda()
                    neg_v = t.LongTensor(neg_v).cuda()
                else:
                    pos_u = t.LongTensor(pos_u)
                    pos_v = t.LongTensor(pos_v)
                    neg_v = t.LongTensor(neg_v)
                self.optimizer.zero_grad()
                loss = self.skip_gram(pos_u, pos_v, neg_v)
                loss.backward()
                self.optimizer.step()
                print(str(datetime.datetime.now()) + '||epoch ' + str(epoch + 1) + '||step ' + str(
                    i + 1) + ' | loss is: ' + str(loss.item()))
        print("----start saving embeddings----")
        self.skip_gram.save(self.use_cuda, self.outputfile)
        print("saving embeddings is success!")


if __name__ == '__main__':
    W2V = Word2Vec('../data/zhihu.txt', './emb.txt')
    W2V.train()

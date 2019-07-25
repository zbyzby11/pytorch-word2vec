"""
skip-gram model
@date 2019.7.25
"""
import json

import torch as t
import torch.nn.functional as F
from torch import nn


class model(nn.Module):
    def __init__(self, embedding_size, embedding_dim):
        """
        模型初始化，输入向量的个数和向量的大小
        :param embedding_size: 输入的emdbedding的长度
        :param embedding_dim: embedding维度
        """
        super(model, self).__init__()
        self.emb_size = embedding_size
        self.emb_dim = embedding_dim
        # u代表word2vec中中心词的向量
        self.u = nn.Embedding(embedding_size, embedding_dim, sparse=True)
        # v代表中心词周围window_size大小的词的向量
        self.v = nn.Embedding(embedding_size, embedding_dim, sparse=True)
        # 随机初始化embedding
        self.init_weight()

    def init_weight(self):
        """
        初始化embedding
        :return: None
        """
        nn.init.xavier_uniform_(self.u.weight.data)
        nn.init.xavier_uniform_(self.v.weight.data)

    def forward(self, pos_u, pos_v, neg_v):
        """
        前向传播
        :param pos_u:正例的中心词
        :param pos_v:正例中心词对应的周围的词
        :param neg_v:负例周围的词
        :return:score得分
        """
        pos_u = self.u(pos_u)
        pos_v = self.v(pos_v)
        pos_score = t.mul(pos_u, pos_v)
        pos_score = pos_score.squeeze()
        pos_score = t.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)

        neg_v = self.v(neg_v)
        neg_score = t.mul(pos_u, neg_v).squeeze()
        neg_score = t.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(neg_score)

        score = -(t.sum(pos_score) + t.sum(neg_score))
        return score

    def save(self, use_cuda, outputfile):
        f = open(outputfile, 'w', encoding='utf8')
        if use_cuda:
            word_embedding = self.u.weight.cpu().data.numpy().tolist()
        else:
            word_embedding = self.u.weight.data.numpy().tolist()
        assert len(word_embedding) == self.emb_size
        f.write(json.dumps(word_embedding, ensure_ascii=False))
        f.close()

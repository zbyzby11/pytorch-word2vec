"""
训练word2vec的模型入口，主函数
@date 2019.7.25
"""
from Word2Vec import Word2Vec

if __name__ == '__main__':
    W2V = Word2Vec('../data/zhihu.txt', './emb.txt')
    W2V.train()


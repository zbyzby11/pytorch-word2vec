"""
测试训练的词向量的效果
@date 2019.7.25
"""
import json
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def similar_word(word, embedding_file, k):
    """
    输出一个词的最相似的k个词
    :param word: 输入的词
    :param embedding_file:词向量保存的文件
    :param k: top-k个词
    :return: list，元素为字典中与word最相似的k个词
    """
    word2id_dict = json.load(open('../dict/word2id_dict.json', 'r', encoding='utf8'))
    id2word_dict = json.load(open('../dict/id2word_dict.json', 'r', encoding='utf8'))
    all_embedding = json.load(open(embedding_file, 'r', encoding='utf8'))
    if word not in word2id_dict:
        raise KeyError
    word_id = int(word2id_dict[word])
    word_embedding = all_embedding[word_id]
    word_embedding = np.array(word_embedding).reshape(1, len(word_embedding))
    all_embedding = np.array(all_embedding)
    # print(word_embedding.shape)
    # print(all_embedding.shape)
    sim = cosine_similarity(word_embedding, all_embedding).flatten()
    # print(sim.shape)
    id_index = np.argsort(-sim)[:k]
    print(id_index)
    for i in id_index:
        print(id2word_dict[str(i)])


if __name__ == '__main__':
    similar_word("电影", "./emb.txt", 10)

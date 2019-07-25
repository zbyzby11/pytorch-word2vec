"""
将data目录中的语料进行预处理
@data 2019.7.24
"""


def create_dict(input_file, min_count):
    """
    将输入的语料对应的每个词进行建立字典，两个字典，一个是词到id
    一个是id到词,还有个列表，列表中是每个正例的元组
    :param input_file:输入文件
    :param min_count:过滤掉词的最小值
    :return: 两个字典
    """
    f = open(input_file, 'r', encoding='utf8')
    word_frequency = dict()
    word2id = dict()
    id2word = dict()
    for line in f:
        line = line.strip().split(' ')
        for w in line:
            if w not in word_frequency:
                word_frequency[w] = 1
            else:
                word_frequency[w] += 1
    f = open(input_file, 'r', encoding='utf8')
    for line in f:
        word_list = line.strip().split(' ')
        for word in word_list:
            if word not in word2id and word_frequency[word] > min_count:
                word2id[word] = len(word2id)

    for word, id in word2id.items():
        id2word[id] = word

    return word2id, id2word


def test():
    """
    测试函数
    :return:None
    """
    x, y = create_dict('../data/zhihu.txt', 4)
    print(len(x))
    print(len(y))


if __name__ == '__main__':
    test()

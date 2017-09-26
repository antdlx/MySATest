from gensim.models import word2vec
import logging

"""
训练word2Vec模型
"""

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
sentences = word2vec.Text8Corpus("corpus.csv")  # 加载语料
model = word2vec.Word2Vec(sentences, size = 400)  # 训练skip-gram模型

# 保存模型，以便重用
model.save("corpus.model")
# 对应的加载方式
# model = word2vec.Word2Vec.load("corpus.model")

# 以一种C语言可以解析的形式存储词向量
model.wv.save_word2vec_format("corpus.model.bin", binary = True)
# 对应的加载方式
# model = word2vec.Word2Vec.load_word2vec_format("corpus.model.bin", binary=True)

import codecs
import re

import jieba

"""
生成训练word2vec的训练语料库
"""

def parseSent(sentence):
    # use Jieba to parse sentences
    seg_list = jieba.cut(sentence)
    output = ' '.join(list(seg_list)) # use space to join them
    return output

# only content is valid
pattern = "<content>(.*?)</content>"
csvfile = codecs.open("corpus.csv", 'w', 'utf-8')
with open("../datas/news.dat", "r",errors='ignore') as txtfile:
    for line in txtfile.readlines():
        m = re.match(pattern, line)
        if m:
            segSent = parseSent(m.group(1))
            csvfile.write("%s" % segSent)
            print(1)
        print(0)
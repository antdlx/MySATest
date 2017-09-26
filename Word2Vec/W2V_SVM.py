import jieba
import numpy as np
from gensim.models import word2vec
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.svm import SVC

from Word2Vec import CleanDatas
import matplotlib.pyplot as plt


def buildVecs(filename, model,debug):
    """
    输入文件名，首先进行数据清洗，然后利用word2vec获得他们的向量化表示，
    :param filename:
    :param model:
    :return:
    """
    with open(filename, 'r',encoding='utf-8',errors='ignore') as f:
        # print txtfile
        content = f.read().replace("\n", '').strip()
        cleaner = CleanDatas.CleanDatas()
        content = cleaner.clean(content)
        seg = jieba.cut(content, cut_all=False)
        s = '-'.join(seg)
        seg_list = s.split('-')
        if len(cleaner.clean_detected_words(seg_list)) > 0:
            seg_list = cleaner.clean_detected_words(seg_list)
        result = []
        for i in seg_list:
            if i in model:
                result.append(model[i])
        array_mean=0.0
        if len(result) != 0:
            array_mean = sum(np.array(result))/len(result)
        else:
            print(debug)
        return array_mean

input_x = []
input_y = []
# 导入模型
model = word2vec.Word2Vec.load('corpus.model')

for i in range(0,3000):
    input_x.append(buildVecs('../datas/neg/neg{0}.txt'.format(i), model,"neg{0}".format(i)))
    input_y.append(0)
for i in range(0,7000):
    input_x.append(buildVecs('../datas/pos/pos{0}.txt'.format(i), model,"pos{0}".format(i)))
    input_y.append(1)

X = np.array(input_x)
Y = np.array(input_y)

#标准化
X = scale(X)

#无监督使用PCA训练X
# pca = PCA(n_components=400)
# pca.fit(X)
#创建图表并指定图表大小
# figsize: w,h tuple in inches
# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.axes([.2, .2, .7, .7])
# plt.plot(pca.explained_variance_, linewidth=2)
# plt.axis('tight')
# plt.xlabel('n_components')
# plt.ylabel('explained_variance_')
# plt.show()

#由图知我们保留前100维的数据
X_reduced = PCA(n_components = 100).fit_transform(X)

#分割训练集和测试集,选取500负和1000正作为测试集
X_reduced_train = X_reduced[0:2500]
X_reduced_train = np.concatenate((X_reduced_train,X_reduced[3000:9000]))
y_reduced_train = Y[0:2500]
y_reduced_train = np.concatenate((y_reduced_train,Y[3000:9000]))
X_reduced_test = X_reduced[2500:3000]
X_reduced_test = np.concatenate((X_reduced_test,X_reduced[9000:10000]))
y_reduced_test = Y[2500:3000]
y_reduced_test = np.concatenate((y_reduced_test,Y[9000:10000]))

#构建SVM模型
clf = SVC(C = 2, probability = True, kernel='linear')

clf.fit(X_reduced_train, y_reduced_train)
print('Test Accuracy: %.5f'% clf.score(X_reduced_test, y_reduced_test))

print("test:")
print(clf.predict(X_reduced_test))
print("value:")
print(y_reduced_test)

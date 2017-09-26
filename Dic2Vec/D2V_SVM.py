import jieba
import numpy as np
from scipy.stats import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC

from Dic2Vec.CleanDatas import CleanDatas


def buildDocList(dic_template,file_postfix,dic_num=0):
    """
    根据输入的目录模板和数量，读取相应目录模板中相应的数量的文件
    :param dic_template: eg."./datas/neg/neg"
    :param dic_num:
    :param file_postfix: 文件的后缀，如txt
    :return:
    """

    results = []
    for i in range(0,dic_num):
        with open("{0}{1}.{2}".format(dic_template,i,file_postfix), 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().replace("\n", '').strip()
            cleaner = CleanDatas()
            content = cleaner.clean(content)
            seg = jieba.cut(content, cut_all=False)
            s = '-'.join(seg)
            seg_list = s.split('-')
            if len(cleaner.clean_detected_words(seg_list)) > 0:
                seg_list = cleaner.clean_detected_words(seg_list)
            s = " ".join(seg_list)
            results.append(s)
    return results

def chi_square(dictionary, matrix, neg_num=0, pos_num=0):
    """
    计算各个特征的卡方值，a,b,c,d分别为观测值，A,B,C,D为预测值，这里因为
    采用的训练语料是非平衡的，比例为3:7，因此A=(a+b)*.7，B=(a+b)*.3，以此类推
            正    负
    包含x1   a     b
    不包含x1 c     d
    通常用计算出的卡方值筛选特征
    :param dictionary: 字典
    :param matrix: 文本的频率矩阵
    :param neg_num: 负文本的数量
    :param pos_num: 正文本的数量
    :return:一个一维数组，包含每个特征X的卡方值，p值越小说明越有区分力，应当选取
    """
    chi_squares = []
    As = []
    Ts = []
    for i in range(0,len(dictionary)):
        a = 0
        b = 0
        for j in range(0,len(matrix)):
            if matrix[j][i] > 0 and j < neg_num:
                b += 1
            if matrix[j][i] > 0 and j >=neg_num:
                a += 1
        c = pos_num - a + 0.01
        d = neg_num - b + 0.01
        A = [a, b, c, d]
        T = [(a+b)*0.7, (a+b)*0.3, (c+d)*0.7, (c+d)*0.3]
        As.append(A)
        Ts.append(T)
        chi_squares.append(stats.chisquare(A, f_exp=T)[1])
    print(As)
    print(Ts)
    return chi_squares

#首先读取并创建正负文本集
corpus_neg = buildDocList("../datas/neg/neg","txt",3000)
corpus_pos = buildDocList("../datas/pos/pos","txt",7000)
#拼接正负文本集为一个总文本集，先负后正
corpus = corpus_neg + corpus_pos

#使用scipy的工具CountVectorizer，根据文本集自动创建字典以及单词频数矩阵
count_vectorizer = CountVectorizer(min_df=1)
term_freq_matrix = count_vectorizer.fit_transform(corpus).toarray()
# print(count_vectorizer.vocabulary_)
# print(corpus)
# print(term_freq_matrix)
#计算字典中每一个特征向量的卡方
chi_square_list = chi_square(count_vectorizer.vocabulary_,term_freq_matrix,3000,7000)
# print(chi_square_list)

#卡方检验降维之后的频数矩阵
term_freq_matrix_after_demetion_reduciton = []
#保留的特征在矩阵中的下角标的集合，如第三、九个特征保留下来，则term_index = [2,8]
term_index = []
#卡方检验中，p值小于这个值的会被保存下来，通常用0.05
INDEX = 0.05
#遍历卡方数组，将需要保存下来的特征下角标保存下来
for i in range(0,len(chi_square_list)):
    if chi_square_list[i] <= INDEX:
        term_index.append(i)
# print(term_index)
print(len(term_index))

#遍历频数矩阵，将相应的需要保留下来的特征保留下来，保存到term_freq_matrix_after_demetion_reduciton（这是一个二维数组）
for i in term_freq_matrix:
    each_term = []
    for j in term_index:
        each_term.append(i[j])
    # if sum(each_term) != 0:   如果删除0则不能确定到底删除的是y=1还是y=0，导致y不好确定
    term_freq_matrix_after_demetion_reduciton.append(each_term)
# print(term_freq_matrix_after_demetion_reduciton)

#使用scipy中的TfidfTransformer自动计算频数矩阵的td-idf值，然后L2标准化，然后用根据td-idf权值制成矩阵
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(term_freq_matrix_after_demetion_reduciton)
tf_idf_matrix = tfidf.transform(term_freq_matrix_after_demetion_reduciton)

# tfidf_vectorizer = TfidfVectorizer(min_df = 1)
# tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
# print(tfidf_matrix.todense())
#分割、构建训练集和测试集
x_matrix = np.array(tf_idf_matrix.todense())
X_reduced_train = np.concatenate((x_matrix[0:2700],x_matrix[3000:9300]))
X_reduced_test = np.concatenate((x_matrix[2700:3000],x_matrix[9300:10000]))
y1 = np.ones(7000)
y0 = np.zeros(3000)
y =  np.concatenate((y0, y1))
y_reduced_train = np.concatenate((y[0:2700], y[3000:9300]))
y_reduced_test = np.concatenate((y[2700:3000], y[9300:10000]))
#使用SVM进行分类，使用线性和函数
clf = SVC(C = 2, probability = True, kernel='linear')
clf.fit(X_reduced_train, y_reduced_train)
print('Test Accuracy: %.5f'% clf.score(X_reduced_test, y_reduced_test))
#Accuracy：0.881
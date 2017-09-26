import jieba
import numpy as np
import tensorflow as tf
from scipy.stats import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tensorflow.contrib import rnn
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
    采用的训练语料是平衡的，所以正负各有50%的概率，因此A=(a+b)/2，以此类推
            正    负
    包含x1   a     b
    不包含x1 c     d
    通常用计算出的卡方值筛选特征
    【PS：注意输入的正负样例数量比必须为1:1】
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
        T = [(a+b)*0.5, (a+b)*0.5, (c+d)*0.5, (c+d)*0.5]
        As.append(A)
        Ts.append(T)
        chi_squares.append(stats.chisquare(A, f_exp=T)[1])
    print(As)
    print(Ts)
    return chi_squares


neg_train_num = 250 #训练集需要的负样例数量
neg_all = 300 #所有负样例的数量
pos_train_num = 250 #训练集需要的正样例的数量
pos_all = 300 #所有正样例的总数量
#首先读取并创建正负文本集
corpus_neg = buildDocList("../datas/neg/neg","txt",neg_all)
corpus_pos = buildDocList("../datas/pos/pos","txt",pos_all)
#拼接正负文本集为一个总文本集，先负后正
corpus = corpus_neg + corpus_pos

#使用scipy的工具CountVectorizer，根据文本集自动创建字典以及单词频数矩阵
count_vectorizer = CountVectorizer(min_df=1)
term_freq_matrix = count_vectorizer.fit_transform(corpus).toarray()
# print(count_vectorizer.vocabulary_)
# print(corpus)
# print(term_freq_matrix)
#计算字典中每一个特征向量的卡方
chi_square_list = chi_square(count_vectorizer.vocabulary_,term_freq_matrix,300,700)
# print(chi_square_list)

#卡方检验降维只有的频数矩阵
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
X_reduced_train = np.concatenate((x_matrix[0:neg_train_num],x_matrix[neg_all:neg_all+pos_train_num]))
X_reduced_test = np.concatenate((x_matrix[neg_train_num:neg_all],x_matrix[neg_all+pos_train_num:neg_all+pos_all]))
# y1 = np.ones(700)
# y0 = np.zeros(300)
# y =  np.concatenate((y0, y1))
# y_reduced_train = np.concatenate((y[0:270], y[300:930]))
# y_reduced_test = np.concatenate((y[270:300], y[930:1000]))

x_mixed_reduced_train = []
y_mixed_train = []
for i in range(0,neg_train_num):
    x_mixed_reduced_train.append(x_matrix[neg_all + i])
    x_mixed_reduced_train.append(x_matrix[i])
    #正，负，ont-hot，这里是正
    tmp_y = [1,0]
    y_mixed_train.append(tmp_y)
    # 正，负，ont-hot，这里是负
    tmp_y = [0, 1]
    y_mixed_train.append(tmp_y)

x_mixed_reduced_test = []
y_mixed_test = []
for i in range(0,neg_all-neg_train_num):
    x_mixed_reduced_test.append(x_matrix[neg_all + pos_train_num + i])
    x_mixed_reduced_test.append(x_matrix[neg_train_num + i])
    #正，负，ont-hot，这里是正
    tmp_y = [1,0]
    y_mixed_test.append(tmp_y)
    # 正，负，ont-hot，这里是负
    tmp_y = [0, 1]
    y_mixed_test.append(tmp_y)

global _current_index
_current_index = 0

def next_batch(batch_size,x_matrix_mixed,y_matrix_mixed):
    global _current_index
    # print(_current_index)
    x = []
    y = []
    if _current_index + batch_size <= len(x_matrix_mixed):
        x = x_matrix_mixed[_current_index:_current_index+batch_size]
        y = y_matrix_mixed[_current_index:_current_index+batch_size]
        _current_index = _current_index + batch_size
    else:
        rest = _current_index+batch_size - len(x_matrix_mixed)
        x = x_matrix_mixed[_current_index:] + x_matrix_mixed[:rest]
        y = y_matrix_mixed[_current_index:] + y_matrix_mixed[:rest]
        _current_index = rest
    return np.array(x),np.array(y)

#==================================以下是LSTM分类代码================================

#学习步伐大小
learning_rate = 0.001
#训练步数
training_steps = 1000
batch_size = 128
#每200步输出一次训练结果
display_step = 20

# Network Parameters
num_input = 1 # 每个单词是一个1*1的ont-hot表示
timesteps = len(term_index) # 每个单词是一个时间步
num_hidden = 128 # 隐藏层
num_classes = 2 # 最后输出的分类，[1,0]正,[0,1]负

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


#开始训练
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = next_batch(batch_size,x_mixed_reduced_train,y_mixed_train)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    # test_label = mnist.test.labels[:test_len]
    test_data = np.array(x_mixed_reduced_test).reshape((-1,timesteps,num_input))
    test_label = np.array(y_mixed_test)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

#Accuracy：0.881
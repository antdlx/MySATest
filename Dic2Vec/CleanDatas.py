import re

"""
用来清晰数据的类
"""

class CleanDatas:
    def clean(self,string):
        """
        清洗特殊符号、标点符号、大写字母、小写字母、数字
        :param string:
        :return:
        """
        string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）；]+|[A-Z]+|[a-z]+|[0-9]", "", string)
        return string

    def clean_detected_words(self,cutted_list):
        """
        清除停用词
        :param cutted_list:
        :return:
        """
        detected_list = []
        with open('../datas/Detected_words_ZH.txt','r',encoding='utf-8',errors='ignore') as f:
            detected_list = f.readlines()

        #下面2行代码的含义是删除最后的\n符号
        for i in range(0,len(detected_list)):
            detected_list[i] = detected_list[i][:-1]

        detected_dics = {}.fromkeys(detected_list,1)
        result = []
        for cutted in cutted_list:
            if cutted not in detected_dics:
                result.append(cutted)
        return result

import os

class DecodeRes:
    """
    将原始数据重新编码成UTF-8
    """
    def decodeNeg2UTF8(self):
        for i in range(0, 7000):
            with open("../datas/pos/pos.{0}.txt".format(i), 'r', errors='ignore') as f:
                content = f.read().strip()
                with open('../datas/pos/pos{0}.txt'.format(i), 'w', encoding='utf-8') as new_file:
                    new_file.write(content)
                    print(content)
            os.remove('../datas/pos/pos.{0}.txt'.format(i))

    def decodePos2UTF8(self):
        for i in range(0, 3000):
            with open("../datas/neg/neg.{0}.txt".format(i), 'r', errors='ignore') as f:
                content = f.read().strip()
                with open('../datas/neg/neg{0}.txt'.format(i), 'w', encoding='utf-8') as new_file:
                    new_file.write(content)
                    print(content)
            os.remove('../datas/neg/neg.{0}.txt'.format(i))
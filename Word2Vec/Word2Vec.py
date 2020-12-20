from bert_serving.client import BertClient
import pandas as pd
import numpy as np
from string import punctuation
import random


# 读取csv文件并初步处理，返回句子列表
def TxtSet():
    # 读取原始的issues.csv文件
    data = np.array(pd.read_csv('.\issues.csv', header=None).values.tolist())
    txt = data[:, 0].tolist()
    newtxt = []
    # 删除每句的标点和前后空格
    for str in txt:
        for i in punctuation:
            str = str.replace(i, '')
        str.strip()
        newtxt.append(str)
    return newtxt


if __name__ == '__main__':
    # 读取初步处理过的文本
    txt = TxtSet()
    # 随机抽取1000条数据
    txt = random.sample(txt, 1000)
    # 打开BERT客户端，会等待服务端启动成功后才能运行，需要手动打开服务端
    bc = BertClient(port=86500, port_out=86501, show_server_config=True)
    # 进行句子转向量
    vec = bc.encode(txt)
    # 将句子向量列表保存成numpy的数据文件
    vec = np.array(vec)
    np.save('word2vec.npy', vec)

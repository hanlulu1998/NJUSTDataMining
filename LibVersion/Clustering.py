import numpy as np
import sklearn.cluster as cluster
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metric
import time


# 返回数据和标签
def DataSet():
    # 使用pandas读取csv文件并转成numpy数组
    data = np.array(pd.read_csv('.\data.csv').values.tolist())
    # 提取X,Y数据
    X = data[:, 1:-1].astype(np.float32)
    Y = data[:, -1].astype(np.float32)
    # 标准化
    StdModel = StandardScaler()
    X = StdModel.fit_transform(X)
    # PCA
    PCAModel = PCA(n_components=34)
    X = PCAModel.fit_transform(X)
    # print(np.nonzero(PCAModel.explained_variance_ratio_ >= 1e-2))
    return X, Y


# 返回文本向量
def TxtSet():
    # 读取文本向量
    txt = np.load('.\word2vec.npy')
    # 标准化
    StdModel = StandardScaler()
    txt = StdModel.fit_transform(txt)
    # PCA
    PCAModel = PCA(n_components=14)
    txt = PCAModel.fit_transform(txt)
    # print(np.nonzero(PCAModel.explained_variance_ratio_ >= 1e-2))
    return txt


# 使用KMeans方法
def useKMeans(X, k):
    model = cluster.KMeans(n_clusters=k)
    yhat = model.fit_predict(X)
    return yhat


# 使用MiniBatchKMeans
def useMiniBatchKMeans(X, k):
    model = cluster.MiniBatchKMeans(n_clusters=k)
    yhat = model.fit_predict(X)
    return yhat


# 使用AGENS
def useAGENS(X, k):
    model = cluster.AgglomerativeClustering(n_clusters=k)
    yhat = model.fit_predict(X)
    return yhat


# 使用OPTICS
def useOPTICS(X, min_samples):
    model = cluster.OPTICS(min_samples=min_samples)
    yhat = model.fit_predict(X)
    return yhat


# 评估函数
def estimate(X, c):
    s1 = metric.silhouette_score(X, c, metric='euclidean')  # 计算轮廓系数
    s2 = metric.calinski_harabasz_score(X, c)  # 计算CH score
    s3 = metric.davies_bouldin_score(X, c)  # 计算 DBI
    return s1, s2, s3


# 运行所有
def runAll(X, k, min_samples=5):
    # 使用KMeans
    print('使用KMeans:')
    start = time.perf_counter()
    c1 = useKMeans(X, k)
    end = time.perf_counter()
    ret = estimate(X, c1)
    print('聚类标签:', np.unique(c1))
    print("轮廓系数:{},CH:{},DBI:{}".format(ret[0], ret[1], ret[2]))
    print('Running time: %s Seconds' % (end - start))

    # 使用MiniBatchKMeans
    print('使用MiniBatchKMeans:')
    start = time.perf_counter()
    c2 = useMiniBatchKMeans(X, k)
    end = time.perf_counter()
    ret = estimate(X, c2)
    print('聚类标签:', np.unique(c2))
    print("轮廓系数:{},CH:{},DBI:{}".format(ret[0], ret[1], ret[2]))
    print('Running time: %s Seconds' % (end - start))

    # 使用AGENS
    print('使用AGENS:')
    start = time.perf_counter()
    c3 = useAGENS(X, k)
    end = time.perf_counter()
    ret = estimate(X, c3)
    print('聚类标签:', np.unique(c3))
    print("轮廓系数:{},CH:{},DBI:{}".format(ret[0], ret[1], ret[2]))
    print('Running time: %s Seconds' % (end - start))

    # 使用OPTICS
    print('使用OPTICS:')
    start = time.perf_counter()
    c4 = useOPTICS(X, min_samples)
    end = time.perf_counter()
    ret = estimate(X, c4)
    print('聚类标签:', np.unique(c4))
    print("轮廓系数:{},CH:{},DBI:{}".format(ret[0], ret[1], ret[2]))
    print('Running time: %s Seconds' % (end - start))


if __name__ == '__main__':
    # 读取数据
    X, Y = DataSet()
    txt = TxtSet()
    # 运行数据集1
    print('运行数据集1:')
    runAll(X, 5)
    # 运行数据集2
    print('运行数据集2:')
    runAll(txt, 8, 14)
    input("程序运行结束，按任意键结束！")

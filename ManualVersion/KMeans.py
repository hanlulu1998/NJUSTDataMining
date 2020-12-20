import numpy as np


# 求两空间点的欧氏距离
def dist2(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# 随机质心
def randCenter(X, k):
    n = X.shape[1]  # 数据列数，即点坐标分量
    centroids = np.zeros((k, n))  # 创建中心点
    for j in range(n):  # 循环所有特征列，获得每个中心点该列的随机值
        valMin = np.min(X[:, j])  # 寻找列的最小值
        valRange = np.float(np.max(X[:, j]) - valMin)  # 寻找列值范围
        centroids[:, j] = valMin + valRange * np.random.rand(k, )  # 获得每列的随机值 一列一列生成
    return centroids


# K-Means方法
def kMeans(X, k):
    m = X.shape[0]  # 数据的个数
    n = X.shape[1]  # 数据的维度
    centroids = randCenter(X, k)  # 随机初始化质心点
    centIndxDist = np.zeros((m, 2))  # 距离矩阵：行数对应每个数据点数目，第一列是与该点距离最近的质心点索引，第二列是距离的平方
    changeFlag = True  # 聚类改变的标志
    while changeFlag:  # 不断循环直到稳定
        changeFlag = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):  # 遍历质心计算与数据点的距离
                dist = dist2(centroids[j, :], X[i, :])
                if dist < minDist:  # 找到与其距离最小的质心点，并在centIndxDist更新
                    minDist = dist
                    minIndex = j
            if centIndxDist[i, 0] != minIndex:  # 只要还在变化就继续更新
                changeFlag = True
            centIndxDist[i, :] = minIndex, minDist ** 2
        for ki in range(k):  # 更新质心
            pInCurrCluster = X[np.nonzero(centIndxDist[:, 0] == ki)]  # 找到所有距离该质心最近的点数据（每一个簇所拥有的所有数据集）
            if len(pInCurrCluster):  # 有数据则进行取平均值
                centroids[ki, :] = np.mean(pInCurrCluster, axis=0)
            else:  # 没有则置0
                centroids[ki, :] = np.zeros((n,))
    return centroids, centIndxDist

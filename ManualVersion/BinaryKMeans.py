from KMeans import *


# 二分k-Means方法
def biKMeans(X, k):
    m = X.shape[0]  # 数据的个数
    centIndxDist = np.zeros((m, 2))  # 距离矩阵：行数对应每个数据点数目，第一列是与该点距离最近的质心点索引，第二列是距离的平方
    centLists = []
    centLists.append(np.mean(X, axis=0))  # 初始质心为全体平均值
    for i in range(m):
        centIndxDist[i, 1] = dist2(centLists[0], X[i, :]) ** 2  # 计算每个点与质心点距离的平方，这里是最初的0簇
    while len(centLists) < k:
        lowestSSE = np.inf  #
        for ki in range(len(centLists)):
            pInCurrCluster = X[np.nonzero(centIndxDist[:, 0] == ki)]  # 找到所有距离该质心最近的点数据（每一个簇所拥有的所有数据集）
            centroidArray, splitIndxDist = kMeans(pInCurrCluster, 2)  # 进行内部进行2-means得到两个质心
            sseSplit = np.sum(splitIndxDist[:, 1])  # 当前簇划分的SSE
            sseNotSplit = np.sum(centIndxDist[np.nonzero(centIndxDist[:, 0] != ki), 1])  # 不再当前簇划分的SSE
            if (sseSplit + sseNotSplit) < lowestSSE:  # 两种方差和小于先前的SSE，说明这种分配方式减小了误差率，可以更新
                bestCentSplit = ki  # 当前簇划分设为最佳
                bestCents = centroidArray  # 当前的划分质心设为最好
                bestIndxDist = splitIndxDist.copy()
                lowestSSE = sseSplit + sseNotSplit  # 更新小的SSE
        bestIndxDist[np.nonzero(bestIndxDist[:, 0] == 1), 0] = len(centLists)  # 2-means返回系数0或1,需要把1换成多出的那个簇数目
        bestIndxDist[np.nonzero(bestIndxDist[:, 0] == 0), 0] = bestCentSplit  # 把0换成最好的那个簇数
        centLists[bestCentSplit] = bestCents[0, :]  # 将最好的那个簇数的质心更新
        centLists.append(bestCents[1, :])  # 添加多出的质心
        centIndxDist[np.nonzero(centIndxDist[:, 0] == bestCentSplit), :] = bestIndxDist  # 更新距离矩阵
    return centLists, centIndxDist

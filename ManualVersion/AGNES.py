import numpy as np


# 求两空间点的欧氏距离
def dist2(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# 求出最小距离
def minDist(Ci, Cj):
    return np.min([dist2(i, j) for i in Ci for j in Cj])


# 求出最大距离
def maxDist(Ci, Cj):
    return np.max([dist2(i, j) for i in Ci for j in Cj])


# 求出平均距离
def avgDist(Ci, Cj):
    return np.mean([dist2(i, j) for i in Ci for j in Cj])


# 找到距离最小的下标
def findMinIndex(M):
    min = np.inf
    x, y = 0, 0
    for i in range(len(M) - 1):
        for j in range(i + 1, len(M[i])):
            if M[i][j] < min:
                min = M[i][j]
                x = i
                y = j
    return x, y, min


# AGNES算法
def AGNES(X, k, distmode='max'):
    if distmode == 'max':
        dist = maxDist
    elif distmode == 'min':
        dist = minDist
    elif distmode == 'avg':
        dist = avgDist
    # 初始化C
    C = []
    # 开始C就是各个点的簇
    for p in X:
        Ci = []
        Ci.append(p.tolist())
        C.append(Ci)
    # 计算所有簇间的最大距离
    m = len(C)
    # 距离矩阵为m*m,上三角
    M = np.zeros((m, m))
    for i in range(len(C) - 1):
        for j in range(i + 1, len(C)):
            M[i][j] = dist(np.array(C[i]), np.array(C[j]))
    # 转置上三角后进行合并矩阵得到M，减少计算量
    q = len(X)
    # 合并更新
    while q > k:
        # 找最近的两个簇
        x, y, min = findMinIndex(M)
        # 合并簇
        C[x].extend(C[y])
        # 删除被合并的,序号自动更改
        C.remove(C[y])
        # 删除距离矩阵的y行，y列
        M = np.delete(M, y, axis=1)
        M = np.delete(M, y, axis=0)
        # 计算合并新的簇，与其他的距离
        # 在x行之前都是行间计算
        for i in range(x):
            M[i][x] = dist(np.array(C[i]), np.array(C[x]))
        for j in range(len(M[x])):
            M[x][j] = dist(np.array(C[j]), np.array(C[x]))
        # 每次簇个数减少1
        q -= 1
    return C

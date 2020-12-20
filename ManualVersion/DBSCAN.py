import random
from scipy.spatial import KDTree


# vistList类用于记录访问列表
class vistList:
    def __init__(self, count=0):
        self.unvisitedList = [i for i in range(count)]
        self.visitedList = []
        self.unvisitedNum = count

    def visit(self, pointId):
        self.visitedList.append(pointId)
        self.unvisitedList.remove(pointId)
        self.unvisitedNum -= 1


def DBSCAN(X, eps, minPts):
    nPoints = X.shape[0]
    # 标记所有对象为unvisited
    vPoints = vistList(count=nPoints)
    # 初始化簇标记列表C，簇标记为 k
    k = -1
    C = [-1 for i in range(nPoints)]
    # 构建KD-Tree，并生成所有距离<=eps的点集合
    kd = KDTree(X)
    while vPoints.unvisitedNum > 0:
        # 随机选择一个unvisited对象p
        p = random.choice(vPoints.unvisitedList)
        # 标t己p为visited
        vPoints.visit(p)
        # N是p的epsilon-邻域点列表
        N = kd.query_ball_point(X[p], eps)
        # 如果p的epsilon邻域至少有MinPts个对象
        if len(N) >= minPts:
            # 创建个一个新簇C，并把p添加到C
            # 对标记列表第p个结点进行赋值
            k += 1
            C[p] = k
            for p1 in N:
                # p1是unvisited
                if p1 in vPoints.unvisitedList:
                    # 标记p1为visited
                    vPoints.visit(p1)
                    # M是p1的epsilon-邻域
                    M = kd.query_ball_point(X[p1], eps)
                    # 如果p1的epsilon-邻域至少有MinPts个点，把这些点去重添加到N
                    if len(M) >= minPts:
                        for i in M:
                            if i not in N:
                                N.append(i)
                    # 如果p1还不是任何簇的成员，把p1添加到c
                    if C[p1] == -1:
                        C[p1] = k
        # 否则标记p为噪声
        else:
            C[p] = -1
    return C

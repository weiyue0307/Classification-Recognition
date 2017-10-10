#coding=UTF8
from numpy import *
import operator

def createDataSet():
    group = array([
        [112,110],
        [128,162],
        [83, 206],
        [142,267],
        [188,184],
        [218,268],
        [234,108],
        [256,146],
        [333,177],
        [350,86],
        [364,237],
        [378,117],
        [409,147],
        [485,130],
        [326,344],
        [387,326],
        [378,435],
        #[434,375]
        [273,230]
    ])
    labels = ['1', '1', '1', '1','1','1','1','1','2','2','2','2','2','2','3','3','3','1']
    return group, labels

def classify0(inX, dataset, labels, k):

    dataSetSize = dataset.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataset
    
    # diffMat就是输入样本与每个训练样本的差值，然后对其每个x和y的差值进行平方运算。
    # diffMat是一个矩阵，矩阵**2表示对矩阵中的每个元素进行**2操作，即平方。
    # sqDiffMat = [[1.0, 0.01],
    #              [1.0, 0.0 ],
    #              [0.0, 1.0 ],
    #              [0.0, 0.81]]
    sqDiffMat = diffMat ** 2
    
    # axis=1表示按照横轴，sum表示累加，即按照行进行累加。
    # sqDistance = [[1.01],
    #               [1.0 ],
    #               [1.0 ],
    #               [0.81]]
    sqDistance = sqDiffMat.sum(axis=1)
    
    # 对平方和进行开根号
    distance = sqDistance ** 0.5
    
    # 按照升序进行快速排序，返回的是原数组的下标。
    # 比如，x = [30, 10, 20, 40]
    # 升序排序后应该是[10,20,30,40],他们的原下标是[1,2,0,3]
    # 那么，numpy.argsort(x) = [1, 2, 0, 3]
    sortedDistIndicies = distance.argsort()
    
    # 存放最终的分类结果及相应的结果投票数
    classCount = {}
    
    # 投票过程，就是统计前k个最近的样本所属类别包含的样本个数
    for i in range(k):
        # index = sortedDistIndicies[i]是第i个最相近的样本下标
        # voteIlabel = labels[index]是样本index对应的分类结果('A' or 'B')
        voteIlabel = labels[sortedDistIndicies[i]]
        # classCount.get(voteIlabel, 0)返回voteIlabel的值，如果不存在，则返回0
        # 然后将票数增1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    
    # 把分类结果进行排序，然后返回得票数最多的分类结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__== "__main__":
    # 导入数据
    dataset, labels = createDataSet()
    inX = [434, 375]
    # 简单分类
    className = classify0(inX, dataset, labels, 3)
    print('The class of test sample is %s' %className)
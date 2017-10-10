#coding=UTF8
from sklearn.model_selection import train_test_split
from numpy import *
import operator
import numpy

def loadDataSet():
    path = 'F:\data\HTRU_2(1).txt'
    data = numpy.loadtxt(path,dtype=float,delimiter=',')    # Read the data
    x, y = numpy.split(data,(8,),axis=1)
    x=x[:,:8]                                # Get the 8 columns of x
    y=y.astype(int)
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=0, train_size=0.8)  # Segment the train-set and test-set
    return train_x,train_y,test_x,test_y


def testHandWritingClass():                   # Calculate the Accuracy(Train)、Accuracy(Test)、Recall、Precision、F1
    train_x,train_y,test_x,test_y = loadDataSet()
    numTestSamples = test_x.shape[0]      # 'numTestSamples' is the number of the test-set
    numTrainSamples=train_x.shape[0]      # 'numTrainSamples' is the number of the train-set
    matchCount=0                          # 'matchCount' is the number of the right prediction of test-set
    bb=0                                  # 'bb' is the number of the right prediction of tran-set
    for i in xrange(numTrainSamples):
        b=kNNClassify(train_x[i],train_x,train_y,8)
        if b==train_y[i]:
            bb+=1
    accuracy1 = float(bb)/numTrainSamples   # Calculate the accuracy for train-set
    print "---Show the result---"
    print 'The classify Accuracy(Train) is: %.2f%%' %(accuracy1 * 100)
    for i in xrange(numTestSamples):
        predict = kNNClassify(test_x[i],train_x,train_y,8)
        if predict==test_y[i]:
            matchCount+=1
    accuracy = float(matchCount)/numTestSamples    # Calculate the accuracy for test-set
    print 'The classify Accuracy(Test) is: %.2f%%' %(accuracy * 100)
    PlusCount=0      # Now we are going to calculate Recall and Precision.
    MinusCount=0     # According to the definitions, I write the following code.
    a=0
    for i in xrange(numTestSamples):
        predict = kNNClassify(test_x[i],train_x,train_y,8)
        if predict==test_y[i]:
            if test_y[i]==1:
                PlusCount+=1
            else:
                MinusCount+=1
        else:
            if predict==1:
                a+=1
    Plus=0
    Minus=0
    for i in xrange(numTestSamples):
        if test_y[i]==1:
            Plus+=1
        else:
            Minus+=1
    Precision=float(PlusCount)/(PlusCount+a)
    Recall=float(PlusCount)/Plus
    print 'The classify Recall is: %.2f%%' %(Recall * 100) # Recall
    print 'The classify Precision is: %.2f%%' %(Precision * 100)  #Precision
    F1=(Recall*Precision*2)/(Recall+Precision)
    print 'The classify F1 is: %.2f%%' %(F1 * 100)  #F1


def kNNClassify(inX, dataset, labels, k):           # KNN algorithm

    dataSetSize = dataset.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataset
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistIndicies = distance.argsort()

    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel[0]] = classCount.get(voteIlabel[0], 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__== "__main__":
    testHandWritingClass()
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy

path = 'F:\data\HTRU_2(1).txt'
data = numpy.loadtxt(path,dtype=float,delimiter=',')  # Read the data
x, y = numpy.split(data,(8,),axis=1)
x=x[:,[2,3]]          # Now I choose the second and third columns in order to get better result.
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.75)
clf = svm.SVC(C=0.8, kernel='rbf', gamma=5, decision_function_shape='ovr',class_weight={1: 3})  # SVM algorithm
clf.fit(x_train, y_train.ravel())
accuracy1=clf.score(x_train, y_train)      # Calculate the accuracy for train-set
print "---Show the result---"
print 'The classify Accuracy(Train) is: %.2f%%' %(accuracy1 * 100)
y_hat = clf.predict(x_train)         # 'y_hat' is the prediction results of train-set
y_hat=y_hat.astype(int)
#print clf.score(x_test, y_test)
y_hat1 = clf.predict(x_test)         # 'y_hat1' is the prediction results of test-set
y_hat1=y_hat1.astype(int)
y_test=y_test.astype(int)
matchCount=0                     # 'matchCount' is the number of the right prediction of test-set
numTestSamples = x_test.shape[0]   # 'numTestSamples' is the number of the test-set
for i in xrange(numTestSamples):
    if y_hat1[i]==y_test[i]:
        matchCount+=1
accuracy = float(matchCount)/numTestSamples  # Calculate the accuracy for test-set
print 'The classify Accuracy(Test) is: %.2f%%' %(accuracy * 100)
PlusCount=0          # Now we are going to calculate Recall, Precision and F1.
MinusCount=0         # According to their definitions, I write the following code.
a=0
b=0
count=0
for i in xrange(numTestSamples):
    if y_hat1[i]==1:
        count+=1
    if y_hat1[i]==y_test[i]:
        if y_test[i]==1:
            PlusCount+=1
        else:
            MinusCount+=1
    else:
        if y_hat1[i]==1:
            a+=1
        else:
            b+=1
Plus=0
Minus=0
for i in xrange(numTestSamples):
    if y_test[i]==1:
        Plus+=1
    else:
        Minus+=1
Recall=float(PlusCount)/Plus
#accuracy2=float(MinusCount)/Minus
Precision=float(PlusCount)/(PlusCount+a)
#accuracy4=float(MinusCount)/(Minus+b)
print 'The classify Recall is: %.2f%%' %(Recall * 100)  # Recall
#print 'The classify accuracy2 is: %.2f%%' %(accuracy2 * 100)
print 'The classify Precision is: %.2f%%' %(Precision * 100)  # Precision
#print 'The classify accuracy4 is: %.2f%%' %(accuracy4 * 100)
F1=(Recall*Precision*2)/(Recall+Precision)
print 'The classify F1 is: %.2f%%' %(F1 * 100)   # F1
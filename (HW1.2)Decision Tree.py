#coding=UTF8
from sklearn import tree
from sklearn.cross_validation import train_test_split
import numpy as np

# Read the data
data=[]
labels=[]
# Write the data into the list
with open('F:\data\HTRU_2(1).txt','r') as f:
    for line in f:
        linelist=line.split(',')
        data.append([float(el) for el in linelist[:-1]])
        labels.append(linelist[-1].strip())
x=np.array(data)
# We can choose some important characters from x's.
# x=x[:,[2,5]]  # Now we choose the third and sixth columns. According to the results, we can find that the accuracy increases, but for the train-set, there is no change and it's still 100.00%
y=np.array(labels)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)# Use cross-validation to segment train-set and test-set, in which train_size is 0.8

# Using entropy of information as criterion, train the Decision Tree. And considering the data's imbalance, I set class_weight='balanced'.
clf=tree.DecisionTreeClassifier(class_weight='balanced',criterion='entropy')
clf.fit(x_train,y_train)
# Write the Decision Tree into the file
with open(r'F:\data\tree.dot','w+') as f:
    f=tree.export_graphviz(clf,out_file=f)

# Ratio reflects the influence of each character.
# Print clf.feature_importances_

anwser1=clf.predict(x_train)      # 'anwser1' is the prediction results of train-set
numTrainSamples=x_train.shape[0]  # 'numTrainSamples' is the number of the train-set
bb=0                              # 'bb' is the number of the right prediction of tran-set
for i in xrange(numTrainSamples):
    if anwser1[i]==y_train[i]:
        bb+=1
accuracy1 = float(bb)/numTrainSamples  # Calculate the accuracy for train-set
#print clf.score(x_train, y_train)   # Using this function, we can also get the same result.
print "---Show the result---"
print "For the train-set:"
print 'The classify Accuracy is: %.2f%%' %(accuracy1 * 100) # Finally, the result I get is 100.00%. But according to the materials on the website, it's a normal phenomenon.

#Then let's try the test-set.
anwser=clf.predict(x_test)        # 'anwser' is the prediction results of test-set
numTestSamples = x_test.shape[0]  # 'numTestSamples' is the number of the test-set
matchCount=0                      # 'matchCount' is the number of the right prediction of test-set
for i in xrange(numTestSamples):
    if anwser[i]==y_test[i]:
        matchCount+=1
accuracy = float(matchCount)/numTestSamples   # Calculate the accuracy for test-set
#print np.mean(anwser==y_test)    # Using either of the two functions, we can also get the same result.
#print clf.score(x_test, y_test)
print "For the Test-set:"
print 'The classify Accuracy is: %.2f%%' %(accuracy * 100)

PlusCount=0                             # Now we are going to calculate Recall and Precision.
MinusCount=0                            # According to the definitions, I write the following code.
a=0
b=0
for i in xrange(numTestSamples):
    if anwser[i]==y_test[i]:
        if y_test[i]=='1':
            PlusCount+=1
        else:
            MinusCount+=1
    else:
        if anwser[i]=='1':
            a+=1
        else:
            b+=1
Plus=0
Minus=0
for i in xrange(numTestSamples):
    if y_test[i]=='1':
        Plus+=1
    else:
        Minus+=1
Recall=float(PlusCount)/Plus
Precision=float(PlusCount)/(PlusCount+a)
print 'The classify Recall is: %.2f%%' %(Recall * 100) #Recall
print 'The classify Precision is: %.2f%%' %(Precision * 100) #Precision
F1=(Recall*Precision*2)/(Recall+Precision)
print 'The classify F1 is: %.2f%%' %(F1 * 100) #F1

#Reference:http://www.cnblogs.com/AlwaysT-Mac/p/6647192.html

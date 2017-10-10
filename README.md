# Classification-Recognition
The first homework of Classification Recognition
王炜越（Weiyue Wang）15307130349

一．	Assignment1：简单分类数据：
Result：（273，230）belongs to 1 class.



In this assignment, I set ‘K’ as ‘3’. And in order to verify the accuracy of the result, I put the test sample (273, 230 , 1)into the train set, and choose some samples such as (434 , 375 , 3) from the original train set as the new test samples. Indeed, I get the results like (434 , 375) belongs to 3 class. So I think the result is right.

Reference： http://blog.csdn.net/niuwei22007/article/details/49703719

二．	Assignment 2：分类任务
In this assignment, I notice that it’s different from the first assignment. Its data is more complex and imbalanced. After reading teacher’s hint and referring to some materials, I decided to accomplish this assignment in three ways. One is to use more classification methods to predict, another is to calculate Accuracy(Train), Accuracy(Test), F1, Precision and Recall, and the last one is to do something with the data. In the following, I will introduce the process of my assignment completion.

Classify Methods：
1.	KNN
After finishing the first assignment, when I saw the second assignment, I tried KNN first. But the problems came, too. In the first place, apart from that we need to read the data from the file, the data in this assignment is different from that in the first assignment. So we need to do something with the data before training. Firstly, we need to separate x and y(labels), and then we need to choose a part of the data as training data set and the rest as testing data set. And in this course, we can notice that the amount of pulse(plus) samples is small. In order to decrease this influence, I set the rate of pulse plus-minus by myself. And I find that 1:1 isn’t the best choice for the result. Besides, considering the imbalance of the data, I divide the data randomly. This part of the code is listed here.
def loadDataSet():
    path = 'F:\data\HTRU_2(1).txt'
    data = numpy.loadtxt(path,dtype=float,delimiter=',')
    x, y = numpy.split(data,(8,),axis=1)
    x=x[:,:8]
    y=y.astype(int)
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=0, train_size=0.7)
    return train_x,train_y,test_x,test_y

In addition, we need to calculate the Accuracy, F1, Precision and Recall. Having understood their meanings, I started coding to calculate them.
 

Accuracy(Train)	Accuracy(Test)	Recall	Precision	F1
97.37%	97.99%	80.29%	92.44%	85.94%
For the KNN method, I find that the results are the same no matter how many times I tried. 

Reference： http://blog.csdn.net/niuwei22007/article/details/49703719

2.	SVM
The second method I used is SVM, whose code is briefer. Using the SVM library, we can use less code to accomplish this assignment. Similarly, considering to diminish the imbalance of the data and according to the teacher’s hint, we need to do something with data and calculate Accuracy, F1, Precision and Recall. Here is the result and code. And in this solution, I set the data’s plus-minus rate.
BUT!!! When I solved all the bugs and run the code happily, a big problem occurred! The Recall and Precision are both 0.00%. Then in order to verify this result, I printed the ‘Pluscount’ in my prediction set and found that it is 0. Besides, I also calculated the Recall and Precision for Minus, and I found that they are both 100.00%. Puzzled, I search the answer on the Internet. And then!!! I found that it is the problem of over-fitting!!!! It’s just the imbalanced data that caused this phenomenon. According to the hints on the Internet, afterwards I did the Feature Analysis using Random Forests, finding that [2,3,0] columns features are more important.

 

As a result, I changed the features chosen in the SVM code and only left [2,3,0] columns features. Then I found that the Precision and Recall (for Plus) increased rapidly. After a lot of experiments, I chose the train_size as 0.75 and the rate of Plus-Minus in the train data set is 1:2, and the result is here.

So F1=(P*R*2)/P+R=82.83%（1：2）([2,3,0])

Then I changed the features again, only left [2,3] columns features, and I found the four results all increased. The result is here.
 
 So F1=(P*R*2)/P+R=87.36%（1：2）([2,3])

Then in the condition that only [2,3] columns features are left, I did five experiments to find the best rate of Plus-Minus. The results are listed.
Plus-MInus	Accuracy	Recall	Precision	F1
1:1	98.03%	81.11%	93.59%	86.90%
1:2	98.03%	84.44%	90.48%	87.36%
1:3	98.03%	85.56%	89.53%	87.50%
1:4	97.92%	86.39%	87.61%	87.00%
1:10	97.32%	87.78%	80.61%	84.04%
And we can find that [2,3] features, train_size 0.75, the rate of Plus-Minus 1:3 are our better choices.
And in this condition, the results are listed as the following.(0.75 , 1:3)
 

Accuracy(Train)	Accuracy(Test)	Recall	Precision	F1
97.81%	98.03%	85.56%	89.53%	87.50%
Also I find that the results are the same no matter how many times I tried.
What’s more, I found that when train_size is bigger, the time used in the process. When the train_size is 0.1, the speed is too high.

Reference：http://www.cnblogs.com/luyaoblog/p/6775342.html

3.	Bayes
In the Bayes Classify Method, the time it takes is extremely short. And in the condition that the rate of train data set and test data set is 0.8, the result is here.
 

Accuracy(Train)	Accuracy(Test)	Recall	Precision	F1	
93.72%	94.05%	87.11%	67.47%	76.04%	
93.00%	93.44%	83.97%	61.54%	71.02%	
93.71%	94.02%	86.08%	61.54%	71.77%	
93.87%	93.07%	83.48%	60.13%	69.00%	
93.81%	93.02%	87.38%	57.61%	69.44%	
93.622%	93.52%	85.604%	61.658%	71.454%	Average

Reference: http://python.jobbole.com/81019/

4.	Decision Tree
In the Decision Tree method, I also set the train_size as 0.8, and its speed is also not bad. And the result is here.
 

Accuracy(Train)	Accuracy(Test)	Recall	Precision	F1	
100.00%	97.15%	86.27%	83.77%	85.00%	
100.00%	96.68%	87.10%	82.10%	84.62%	
100.00%	97.15%	85.12%	84.62%	84.87%	
100.00%	96.70%	85.71%	80.99%	83.29%	
100.00%	96.31%	82.62%	80.33%	81.46%	
100.00%	96.798%	85.364%	82.362%	83.848%	Average

     But what puzzles me is that the accuracy of the train-set is 100.00%, which is thought as normal on the website. And I have tried many times, but the result is still 100.00%.

Reference: http://www.cnblogs.com/AlwaysT-Mac/p/6647192.html
Reference: http://blog.csdn.net/sysu_cis/article/details/51842736

Comparison:
	Accuracy(Train)	Accuracy(Test)	Recall	Precision	F1
KNN	97.37%	97.99%	80.29%	92.44%	85.94%
SVM	97.81%	98.03%	85.56%	89.53%	87.50%
Bayes	93.622%	93.52%	85.604%	61.658%	71.454%
Decision Tree	100%	96.798%	85.364%	82.362%	83.848%
And the time Bayes uses is least.


This is the end of my report, and I appreciate it so much that you are reading until this place!! Thank you!!! 

END

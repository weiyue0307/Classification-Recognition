import csv
import random
import math

def loadCsv(filename):                          # Read the data
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, splitRatio):         # Split the data
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):            # In the following are the algorithms of Decision-Tree
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():                              # 'main' function
	filename = 'F:\data\HTRU_2.csv'      # Read the data
	splitRatio = 0.8                     # Set the train-set as 80% of the data
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	summaries = summarizeByClass(trainingSet)
	predictions = getPredictions(summaries, testSet)      # 'predictions' is the prediction-results of test-set
	prediction=getPredictions(summaries,trainingSet)      # 'prediction' is the prediction-results of train-set
	accuracy = getAccuracy(trainingSet, prediction)       # Calculate the accuracy for train-set
	#print('Accuracy: {0}%').format(accuracy)
	print "---Show the result---"
	print 'The classify Accuracy(Train-set) is: %.2f%%' %(accuracy ) #precision
	#accuracy = getAccuracy(testSet, predictions)
	#print('Accuracy: {0}%').format(accuracy)
	matchCount=0                              # 'matchCount' is the number of the right prediction of test-set
	numTestSamples=len(testSet)           # 'numTestSamples' is the number of the test-set
	for i in xrange(numTestSamples):
		if predictions[i]==testSet[i][8]:
			matchCount+=1
	accuracy = float(matchCount)/numTestSamples  # Calculate the accuracy for test-set
	print 'The classify Accuracy(Test-set) is: %.2f%%' %(accuracy * 100) #precision
	PlusCount=0                       # Now we are going to calculate Recall, Precision and F1.
	MinusCount=0                      # According to their definitions, I write the following code.
	a=0
	for i in xrange(numTestSamples):
		if predictions[i]==testSet[i][8]:
			if testSet[i][8]==1:
				PlusCount+=1
			else:
				MinusCount+=1
		else:
			if predictions[i]==1:
				a+=1
	Plus=0
	Minus=0
	for i in xrange(numTestSamples):
		if testSet[i][8]==1:
			Plus+=1
		else:
			Minus+=1
	Recall=float(PlusCount)/Plus
	Precision=float(PlusCount)/(PlusCount+a)
	print 'The classify Recall is: %.2f%%' %(Recall * 100) #recall
	print 'The classify Precision is: %.2f%%' %(Precision * 100) #precision
	F1=(Recall*Precision*2)/(Recall+Precision)
	print 'The classify F1 is: %.2f%%' %(F1 * 100) #F1


main()
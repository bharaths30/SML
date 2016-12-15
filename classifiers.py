import math
import random
import bisect
import collections
from numpy.random import choice


from sklearn import linear_model




smalleta = 0.00000000001

class LogisticRegression():
	def __init__(self, dataset = [], eta = smalleta):
		self.lr = linear_model.LogisticRegression()
		y = []
		x = []
		for i in dataset:
			y.append(i[0])
			x.append(i[1])
		#print(y)
		self.lr.fit(x,y)

	def getCoefficients(self):
		return self.lr.coef_.tolist()[0]

	def getClass(self, X):
		return self.lr.predict(X)[0]

	

class Boosting():
	def __init__(self, dataset):
		self.dataset = dataset
		self.weightsArr = []
		self.alphas = []
		self.lrWeights = []

	def getErrorRate(self, actualClassLabels, predictedClassLabels):
		count = 0
		weights = self.weightsArr[-1]
		for index in range(len(actualClassLabels)):
			if actualClassLabels[index] != predictedClassLabels[index]:
				count += float(weights[index])
		return float(count)

	def getAlphaT(self, errrorRate):
		try:
			return float(0.5 * math.log(float(1 - errrorRate) / float(errrorRate)))
		except:
			print (errrorRate)

	def getNewWeights(self, alphaT, predictedClassLabels):
		normalizer = 0.0
		oldWeights = self.weightsArr[-1]
		tempWeights = []
		for index in range(len(predictedClassLabels)):
			weight = float(oldWeights[index]) * float(math.exp(float(-1) * float(self.dataset[index][0]) * float(predictedClassLabels[index]) * float(alphaT)))
			normalizer += float(weight)
			tempWeights.append(float(weight))
		newWeights = [float(x) / float(normalizer) for x in tempWeights]
		self.weightsArr.append(newWeights)

	def getNewTrainingSet(self):
		indexArr = []
		for i in range(len(self.dataset)):
			indexArr.append(i)
		newIndexArr = choice(indexArr, len(self.dataset), self.weightsArr[-1])
		newTrainingSet = []
		for i in newIndexArr:
			newTrainingSet.append(self.dataset[i])

		return newTrainingSet

	def getWeights(self):
		return self.lrWeights

	def trainAlphas(self):
		testset = list(self.dataset)
		trainingset = list(self.dataset)

		weights = []
		for i in range(len(testset)): 
			weights.append(float(1.0 / len(testset)))
		self.weightsArr.append(weights)
		
		for k in range(8):
			predictedClassLabels = []
			actualClassLabels = []
			print ('Training LR', k + 1)
			lr = LogisticRegression(trainingset, smalleta)
			for example in testset:
				val = lr.getClass(example[1])
				actualClassLabels.append(example[0])
				predictedClassLabels.append(val)
			errrorRate = self.getErrorRate(actualClassLabels, predictedClassLabels)
			print (errrorRate)
			if errrorRate == 0 or errrorRate == 1:
				break;
				print (ls.getCoefficients)
			self.lrWeights.append(lr.getCoefficients())
			self.alphas.append(self.getAlphaT(errrorRate))
			self.getNewWeights(self.alphas[-1], predictedClassLabels)
			trainingset = self.getNewTrainingSet()
			#If only 1 class is sent
			classValY=[]
			for each in trainingset:
				classValY.append(each[0])
			if len(set(classValY))==1:
				print (classValY)
				break	
		return self.alphas
			
	def getClass(self, X, alphas, coefficients):
		total = 0.0
		for k in range(10):
			if k < len(alphas):
				total += float(alphas[k]) * float(getClassGivenCoefficients(X, coefficients[k]))
		if total > 0:
			return 1
		else:
			return -1

def getClassGivenCoefficients(X, coefficients):
		total = 0.0
		for feature in range(len(X)):
			total += float(X[feature]) * float(coefficients[feature])
		if total <= 0:
			return -1
		else:
			return 1
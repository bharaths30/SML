import math
import random
import bisect
import collections
from numpy.random import choice

smalleta = 0.00000000001

class LogisticRegression():
	def __init__(self, dataset = [], eta = smalleta):
		self.dataset = dataset
		self.eta = float(eta)
		self.coefficientList = []
		coefficients = []
		for i in self.dataset[0][1]:
			coefficients.append(0)
		coefficients.append(0)
		self.coefficientList.append(coefficients)

	def getProbabilityY1givenXW(self, data, coefficients):
		try:
			val = float(coefficients[0])
			for index in range(len(data[1])):
				val += float(coefficients[index + 1]) * float(data[1][index])
				K=float(math.exp(val * data[0]))
			return K/(1.0 + K)
		except:
			print (val * data[0])

	def getNextCoefficients(self, coefficients):
		probabilityDifferenceList = []
		for data in self.dataset:
			probabilityDifferenceList.append(float(data[0]) - self.getProbabilityY1givenXW(data, coefficients))
		
		newCoefficients = []
		newCoefficients.append(float(coefficients[0]) + float(self.eta * sum(probabilityDifferenceList)))
		for coeffIndex in range(len(coefficients[1:])):
			total = 0.0
			for probabilityIndex in range(len(probabilityDifferenceList)):
				total += float(self.dataset[probabilityIndex][1][coeffIndex]) * float(probabilityDifferenceList[probabilityIndex])
			newCoefficients.append(float(coefficients[coeffIndex + 1]) + float(self.eta * total))

		return newCoefficients

	def trainCoefficients(self):
		for i in range(len(self.dataset)):
			nextCoeffs = self.getNextCoefficients(self.coefficientList[-1])
			self.coefficientList.append(nextCoeffs)
			

	def getCoefficients(self):
		return list(self.coefficientList[-1])
		
	def getClass(self, X):
		coefficients = self.getCoefficients()
		total = coefficients[0]
		for feature in range(len(X)):
			total += float(X[feature]) * float(coefficients[feature + 1])
		if total <= 0:
			return -1
		else:
			return 1

	

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
		return float(0.5 * math.log(float(1 - errrorRate) / float(errrorRate)))

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
		
		for k in range(4):
			predictedClassLabels = []
			actualClassLabels = []
			print ('Training LR', k + 1)
			lr = LogisticRegression(trainingset, smalleta)
			lr.trainCoefficients()
			for example in testset:
				val = lr.getClass(example[1])
				actualClassLabels.append(example[0])
				predictedClassLabels.append(val)
			errrorRate = self.getErrorRate(actualClassLabels, predictedClassLabels)
			if errrorRate == 0 or errrorRate == 1:
				break;
				print (ls.getCoefficients)
			self.lrWeights.append(lr.getCoefficients())
			self.alphas.append(self.getAlphaT(errrorRate))
			self.getNewWeights(self.alphas[-1], predictedClassLabels)
			trainingset = self.getNewTrainingSet()
			
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
		total = coefficients[0]
		for feature in range(len(X)):
			total += float(X[feature]) * float(coefficients[feature + 1])
		if total <= 0:
			return -1
		else:
			return 1
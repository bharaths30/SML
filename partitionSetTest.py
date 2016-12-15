#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      sbhar
#
# Created:     05/11/2016
# Copyright:   (c) sbhar 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy,csv,sys,CSVFileParser
import xml.etree.cElementTree as ET
from classifiers import Boosting
import MSE
import time


def partitionSet(featureSet,usefulVotes):
    #Upper set < mean aka left
    upperSet=[] #Sets with values lower than mean/median
    upperSetValues=[]#Class 0
    lowerSet=[] #Sets with values higher than mean/median
    lowerSetValues=[] #Class 1
    splitVal=numpy.mean(usefulVotes) #Split based on
    numberOfExamples=len(usefulVotes)
    i=0
    while i<numberOfExamples:
        if usefulVotes[i]<splitVal:
            upperSet.append(featureSet[i])
            upperSetValues.append(usefulVotes[i])
        else:
            lowerSet.append(featureSet[i])
            lowerSetValues.append(usefulVotes[i])
        i+=1
    return (upperSet,upperSetValues,lowerSet,lowerSetValues)

def runTraining(featureSet,usefulVotes,discreteClasses):
    #Boosting with LR,NB,KNN
    #return (boostingParameters,LRParameters,NBParameters)
    return (featureSet[0],featureSet[0],featureSet[0])

def setDiscreteClassValues(featureSet,usefulVotes):
    discreteClasses=[]
    splitVal=numpy.mean(usefulVotes)
    #print splitVal
    for each in usefulVotes:
        if each<splitVal:
            discreteClasses.append(-1)
        else:
            discreteClasses.append(1)
    return discreteClasses

def modifyFeatureSet(featureSet,ClassValues):
    retFeatureSet=[]
    for i in range(0,len(featureSet)):
        retFeatureSet.append((ClassValues[i],featureSet[i]))
    return retFeatureSet

def predictUsefulVotes(node,featureSetWhole,usefulVotes,example):
    discreteClassValues=setDiscreteClassValues(featureSetWhole,usefulVotes)
    featureSet=modifyFeatureSet(featureSetWhole,discreteClassValues)
    #predictedClass=classify(featureSet,usefulVotes,discreteClassValues,example,eval(node[0].text),eval(node[1].text),eval(node[2].text)) #node[0] - Boosting params, node[1]-LR weights, node[2]-NB
    #print ("Set values",len(set(discreteClassValues)))
    #time.sleep(1)
    if len(set(discreteClassValues)) <= 1:
        #print (usefulVotes)
        return usefulVotes[0]
    b=Boosting(featureSet)
    alphas=eval(node[0].text)
    LRWeights=eval(node[1].text)
    predictedClass=b.getClass(example,alphas,LRWeights)
    #print len(node)
    if len(node)>2:#3:
        #node[4] left and node[5] right
        (leftSet,leftSetUsefulVotes,rightSet,rightSetUsefulVotes)=partitionSet(featureSetWhole,usefulVotes)  #Upper set < mean aka left
        if predictedClass==-1:
            return predictUsefulVotes(node[2],leftSet,leftSetUsefulVotes,example)
        else:
            return predictUsefulVotes(node[3],rightSet,rightSetUsefulVotes,example)
    else:
        if predictedClass==-1:
            return usefulVotes[0]
        else:
            return usefulVotes[1]
        """knn=KNN(featureSet)
        print featureSet
        return knn.getClass(example,1)"""
        #return usefulVotes[0]

def loadParametersXML(XMLFileName):
    tree=ET.parse(XMLFileName)#"trainingParameters.xml"
    root=tree.getroot()
    return root

def predictExample(example,featureSet,usefulVotes,root):
    #root=loadParametersXML()
    #(featureSet,usefulVotes)=parseTrainingCSV() #Parse the denormalized csv file
    noOfUsefulVotes=predictUsefulVotes(root,featureSet,usefulVotes,example)
    return noOfUsefulVotes

def BoostingTest(filename,noOfTrainingExamples,XMLFileName):
    import partitionSet
    partitionSet.TrainDataSet(filename,noOfTrainingExamples,XMLFileName)
    (featureSetTrain,usefulVotesTrain)=CSVFileParser.parseTrainingCSV(filename,noOfTrainingExamples)
    (featureSet,usefulVotes)=CSVFileParser.parseTestingCSV(filename,0,noOfTrainingExamples)
    root=loadParametersXML(XMLFileName)
    #print len(featureSet)
    predictedValues=[]
    for each in featureSet:
        val=predictExample(each,featureSetTrain,usefulVotesTrain,root)
        predictedValues.append(val)
        #print "Value:"+str(val)
    #print "Predited Class Values",predictedValues
    #print "Actual values",usefulVotes
    return MSE.mse(predictedValues,usefulVotes)


"""if __name__ == '__main__':
    main()"""

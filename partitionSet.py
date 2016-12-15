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
minThreshold=2
from classifiers import Boosting
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
    #print featureSet
    for i in range(0,len(featureSet)):
        retFeatureSet.append((ClassValues[i],featureSet[i]))
    return retFeatureSet

def trainTree(node,featureSetWhole,usefulVotes):
    #print featureSet
    discreteClassValues=setDiscreteClassValues(featureSetWhole,usefulVotes)
    #print (discreteClassValues,usefulVotes)
    
    featureSet=modifyFeatureSet(featureSetWhole,discreteClassValues)
    print (len(featureSet))
    count = 0
    for i in featureSet:
        count += i[0]
    print (count,len(featureSet))
    if len(featureSet) == count or len(featureSet) == (-1 * count):
        #print ("1 class")
        #time.sleep(10)
        return
    #boostingParameters=runTraining(featureSet,usefulVotes,discreteClassValues)
    b=Boosting(featureSet)
    boostingParameters=b.trainAlphas()
    LRWeights=b.getWeights()
    #print boostingParameters
    ET.SubElement(node,"BoostingAlphas").text=str(boostingParameters)
    ET.SubElement(node,"LRWeights").text=str(LRWeights)
    #ET.SubElement(node,"NBParameters").text=str(NBParameters)
    if len(featureSet)>minThreshold:
        (leftSet,leftSetUsefulVotes,rightSet,rightSetUsefulVotes)=partitionSet(featureSetWhole,usefulVotes)  #Upper set < mean aka left
        if len(leftSet)==len(featureSet) or len(rightSet)==len(featureSet):
            return
        leftChild=ET.SubElement(node,"LeftChild")
        trainTree(leftChild,leftSet,leftSetUsefulVotes)
        rightChild=ET.SubElement(node,"RightChild")
        trainTree(rightChild,rightSet,rightSetUsefulVotes)
    return



def TrainDataSet(filename,noOfTrainingExamples,XMLFileName):
    (featureSet,usefulVotes)=CSVFileParser.parseTrainingCSV(filename,noOfTrainingExamples) #Parse the denormalized csv file #Training Set File, No. of Training Examples
    root=ET.Element("root")
    trainTree(root,featureSet,usefulVotes)
    tree = ET.ElementTree(root)
    tree.write(XMLFileName)#XML File Path

"""def main():
    TrainDataSet()

if __name__ == '__main__':
    main()"""

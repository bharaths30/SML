#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      sbhar
#
# Created:     12/11/2016
# Copyright:   (c) sbhar 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os,sys,csv,CSVFileParser,gc
import numpy as np
import MSE
#regParam=62.0

"""def parseTestingCSV():
    data = []
    h=[]
    t=[]
    #csv_data = csv.reader(file("C:\\D-Drive\\studies\\MCS\\SML\\Project\\yelp_data\\yelp_training_set\\training_data_dummy.csv"))
    csv_data = csv.reader(file(sys.argv[1]))
    i=0
    for row in csv_data:
        i=i+1
        if(i<800):
            continue
        if(i>900):
            break
        data.append(row)
    for i in range(0,len(data)):
        features=[]
        for j in range(1,len(data[i])-1):
            #h.append(data[i][j])
            features.append(float(data[i][j]))
        h.append(features)
        t.append(float(data[i][len(data[i])-1]))
    return (h,t)


def parseTrainingCSV():
    data = []
    h=[]
    t=[]
    #csv_data = csv.reader(file("C:\\D-Drive\\studies\\MCS\\SML\\Project\\yelp_data\\yelp_training_set\\training_data_dummy.csv"))
    csv_data = csv.reader(file(sys.argv[1]))
    i=0
    for row in csv_data:
        i=i+1
        if(i==1):
            continue
        if(i>500):
            break
        data.append(row)
    for i in range(0,len(data)):
        features=[]
        for j in range(1,len(data[i])-1):
            #h.append(data[i][j])
            features.append(float(data[i][j]))
        h.append(features)
        t.append(float(data[i][len(data[i])-1]))
    return (h,t)"""

def appendOneToBasis(BasisH):
    for each in BasisH:
        each.append(1)
    return BasisH

def trainLinearRegression(BasisH,usefulVotes,regParam):
    BasisH=appendOneToBasis(BasisH)
    #print BasisH,usefulVotes
    H=np.array(BasisH)
    t=np.array(usefulVotes)
    Ht=np.transpose(H)
    #w=(H'H)-1H't
    HtH=np.dot(Ht,H)
    size=len(HtH)
    regularizer=regParam*np.array(np.identity(size))
    Ainv=np.linalg.pinv(np.add(HtH,regularizer))
    b=np.dot(Ht,t)
    weights=np.dot(Ainv,b)
    return weights

def predictUsingLinearRegression(BasisH,w):
    return np.dot(BasisH,w)

def LinearRegression(datasetFile,noOfTrainingExamples,regParam):
    #trainDataSet="C:\\D-Drive\\studies\\MCS\\SML\\Project\\yelp_data\\yelp_training_set\\training_data_dummy.csv"#sys.argv[1]
    #testingDataSet="C:\\D-Drive\\studies\\MCS\\SML\\Project\\yelp_data\\yelp_training_set\\training_data_dummy.csv"#sys.argv[2]
    (hTrain,tTrain)=CSVFileParser.parseTrainingCSV(datasetFile,noOfTrainingExamples)
    w=trainLinearRegression(hTrain,tTrain,regParam)
    #print w
    (hTest,tTest)=CSVFileParser.parseTestingCSV(datasetFile,30000,500)
    predictedValuesTest=predictUsingLinearRegression(appendOneToBasis(hTest),w)
    predictedValuesTrain=predictUsingLinearRegression(hTrain,w)
    #print "Predicted Values",predictedValues
    #print "Actual Values",tTest
    #print "MSE Test",MSE.mse(predictedValues,tTest)   
    #print (str(noOfTrainingExamples)+","+str(regParam)+","+str(MSE.mse(predictedValuesTest,tTest))+","+str(MSE.mse(predictedValuesTrain,tTrain)))
    gc.collect()
    return MSE.mse(predictedValuesTest,tTest)

#print ("No. of Training Examples,Lamda,Test Error,Train Error")
#LinearRegression(sys.argv[1],int(sys.argv[2]),float(sys.argv[3]))
    


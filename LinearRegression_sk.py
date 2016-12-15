from sklearn import linear_model
import CSVFileParser,MSE,sys
datasetFile=sys.argv[1]
noOfTrainingExamples=int(sys.argv[2])
(hTrain,tTrain)=CSVFileParser.parseTrainingCSV(datasetFile,noOfTrainingExamples)
lr = linear_model.LinearRegression()
lr.fit(hTrain,tTrain)
print (lr.coef_)
(hTest,tTest)=CSVFileParser.parseTestingCSV(datasetFile,30000,100)
predictedValues=lr.predict(hTest).tolist()
print (predictedValues)
rmse=MSE.mse(tTest,predictedValues)
print (rmse)

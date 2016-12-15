import linear_regression,partitionSetTest
import sys
xAxis = []
yAxis1 = []
yAxis2 = []

for i in range(1,7):
	xAxis.append(i*1000)
	yAxis1.append(linear_regression.LinearRegression(sys.argv[1],i*1000,1000.0))
	#yAxis2.append(partitionSetTest.BoostingTest(sys.argv[1],i*1000,sys.argv[2]))

print(xAxis,yAxis1)

import matplotlib.pyplot as plt

def plot(xAxis, yAxis2):
	plt.title('Length of Training Data VS Train Error 1 Gram - Iterative Boosting')
	p1=plt.plot(xAxis, yAxis2, label="Iterative Boosting")
	#p2=plt.plot(xAxis, yAxis1, label="Linear Regression")
	plt.axis([0, 7000, 0, 3])
	plt.ylabel('Error Count')
	plt.xlabel('Length of Training Data')
	#plt.legend(loc='upper right', shadow=True)
	plt.show()

xAxis=[1000, 2000, 3000, 4000, 5000, 6000]
#yAxis1=[1.8875579716237876, 1.8275166234036069, 1.8115916685776665, 1.8121554261324897, 1.8106323640170532, 1.810848842173118]
yAxis2=[2.320129306741329, 1.9489740891043166, 1.6865151447091524, 1.6811454428454429, 2.1000476185077326, 2.0411189741577207]

plot(xAxis, yAxis2)



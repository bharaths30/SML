import matplotlib.pyplot as plt

def plot(xAxis, yAxis1, yAxis2):
	plt.title('Length of Training Data VS Error count Random Forest 3 Gram')
	p1=plt.plot(xAxis, yAxis2, label="PCA")
	p2=plt.plot(xAxis, yAxis1, label="Random Forest")
	plt.axis([0, 7000, 0, 20])
	plt.ylabel('Error Count')
	plt.xlabel('Length of Training Data')
	plt.legend(loc='upper right', shadow=True)
	plt.show()

xAxis=[1000, 2000, 3000, 4000, 5000, 6000]
yAxis1=[3.378338889557205, 3.103021513778153, 3.0093002880325677, 3.5716224640838097, 4.953677639460265, 5.453669862584265] #RF
yAxis2=[7.870077747550846, 5.899355491032969, 5.768487592237363, 7.905126056178475, 11.27296945437897, 8.299411998506766] #PCA

plot(xAxis, yAxis1, yAxis2)

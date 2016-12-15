def mse(list1,list2):
	error=0.0
	for i in range(0,len(list1)):
		v=float((list1[i]-list2[i])**2)
		error+=v
	meanerror=float(error/len(list1))
	return meanerror**0.5
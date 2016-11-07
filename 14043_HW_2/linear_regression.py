import numpy as np
import matplotlib.pyplot as plt
import re
import copy
f = ''

def parseFile(fileName):
	file = open(fileName, 'rb')
	dataSet = []
	for line in file:
		line = line.replace(' ', ',').replace('\t',',')
		data = line.split(',')
		for i in range(len(data)):
			if (data != '\n'):
				data[i] = float(data[i].strip())
		dataSet.append(data)
	return np.array(dataSet)

def MeanSquareError(x, y, theta):
	n = len(x)
	m = len(x[0])
	
	y_h = x.dot(theta)
	return (np.linalg.norm(y_h - y) ** 2) / n

def scatter_plot(x, y, y_predicted, title):
	plt.scatter(x, y)
	plt.scatter(x, y_predicted, c = 'green')
	plt.title(title)
	plt.show()

def plot(X, Y, title, handle, xlabel, ylabel):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	p, = plt.plot(X, Y, c = 'blue', label = handle)
	for xy in zip(X, Y):
		ax.annotate('(%s, %s)' % xy, xy = xy, textcoords = 'data')

	plt.legend(handles = [p])
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

def multiplot(X, Y, title, handle, xlabel, ylabel, colors):
	l = []
	for i in range(len(X)):
		p, = plt.plot(X[i], Y[i], c = colors[i], label = handle[i])
		l.append(p)
	
	plt.legend(handles = l)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

def linearKernel(X):
	n = len(X)
	x = np.c_[np.ones(n), X[:,:-1]]
	y = X[:,-1]
	return x, y

def polynomialKernel(X, degree):
	n = len(X)
	z = np.array(X[:,0])
	x = np.ones(n)
	x.shape = (n, 1)
	for i in range(degree):
		x = np.c_[x, np.power(z, i + 1)]
	y = X[:,1]
	return x, y

def gaussianKernel(X):
	n = len(X)
	y = X[:,1]
	variance = np.var(y)
	mean = np.ones(n) * X[:,0].mean()
	x = np.array(X[:,0])
	x = (x - mean)
	for i in range(len(x)):
		x[i] = np.exp(-x[i] * x[i] / (2 * variance))
	x = np.c_[np.ones(n), x]
	return x, y

def gradient_descent(x, y, alpha, max_itr, initial_del, param = 5):
	m, n = len(x[0]), len(x)
	MSE, q = np.zeros(param), np.zeros(param)
	theta = np.zeros(m)
	
	if param == 5:
		for quantum in range(param):
			z = x[0 : n / 2 + quantum * (n / 10)]
			w = y[0 : n / 2 + quantum * (n / 10)]
			for itr in range(max_itr):
				h = z.dot(theta)
				theta = theta - alpha * (np.transpose(h - w).dot(z) + initial_del * theta)
			cost = MeanSquareError(x, y, theta)
			MSE[quantum] = cost
			q[quantum] = 50 + quantum * 10
		return q, MSE, theta
	else:
		theta = np.zeros(x.shape[1])
		for itr in range(max_itr):
			h = x.dot(theta)
			theta = theta - alpha * (np.transpose(h - y).dot(x) + initial_del * theta)
		cost = MeanSquareError(x, y, theta)
		return 0, cost, theta

def ten_fold_cross_validation(X, max_itr, initial_del):
	n = len(X)
	np.random.shuffle(X)
	x, y = linearKernel(X)
	val = {'seeds_dataset.txt': 8.75e-10, 'iris.data': 8.5e-10, 'AirQualityUCI.csv': 5e-13}
	alpha = val[f]

	data_x = np.array_split(x, 10)
	data_y = np.array_split(y, 10)
	mean = np.zeros(10)
	variance = np.zeros(10)
	print 'Ten-fold cross validation started.'
	for i in range(10):
		print 'Iteration ' + str(i + 1)
		test = np.c_[data_x[i], data_y[i]]
		temp = []
		for j in range(10):
			if i != j:
				temp.append(np.c_[data_x[i], data_y[i]])
		training = temp[0]
		for j in range(1, len(temp)):
			training = np.r_[training, temp[j]]
		x, y = linearKernel(training)
		q, MSE, theta = gradient_descent(x, y, alpha, max_itr, initial_del, 1)
		print 'MSE:', MSE
		y_predicted = x.dot(theta)
		mean[i] = (y - y_predicted).mean()
		variance[i] = (np.var(y - y_predicted))**0.5
	m, v = mean.mean(), variance.mean()
	return m, v

def linear_regression(X, phi, max_itr, initial_del):
	theta, x, y = [], [], []
	x1, y1, x2, y2 = [], [], [], []	# For phi = 2 or 3
	m, n, alpha, alpha1, alpha2 = 0, 0, 0, 0, 0
	MSE, q = np.zeros(5), np.zeros(5)
	
	title = f + ', Data used(%) vs MSE, ' + str(max_itr) + ' iterations'
	handle = ''

	if phi == 4:
		x, y = polynomialKernel(X, 2)
		x1, y1 = polynomialKernel(X, 3) 
		x2, y2 = gaussianKernel(X)
		val = {'sph.txt': (5e-10, 5e-14), 'lin.txt': (5e-7, 5e-10)}
		val1 = {'sph.txt': 5e-4, 'lin.txt': 1e-3}
		alpha = val[f][0]
		alpha1 = val[f][1]
		alpha2 = val1[f]

		q, MSE, theta = gradient_descent(x, y, alpha, max_itr, initial_del)
		q1, MSE1, theta = gradient_descent(x1, y1, alpha1, max_itr, initial_del)
		q2, MSE2, theta = gradient_descent(x2, y2, alpha2, max_itr, initial_del)
		l_q = [q, q1, q2]
		l_mse = [MSE, MSE1, MSE2]
		handle = ['Polynomial kernel with degree 2', 'Polynomial kernel with degree 3', 'Gaussian kernel']
		xlabel = 'Percentage of data used(%)'
		ylabel = 'Mean square error'
		colors = ['black', 'red', 'blue']
		multiplot(l_q, l_mse, title, handle, xlabel, ylabel, colors)
		return -1

	if phi == 1:
		x, y = linearKernel(X)
		handle = 'Linear kernel'
		val = {'sph.txt': 2.5e-6, 'lin.txt': 2.5e-4, 'seeds_dataset.txt': 8.75e-6, 'iris.data': 8.5e-6, 'AirQualityUCI.csv': 8.5e-13}
		alpha = val[f]

	elif phi == 2:
		x, y = polynomialKernel(X, 2)
		handle = 'Polynomial kernel with best-fit degree 2'
		val = {'sph.txt': 5e-10, 'lin.txt': 5e-7}
		alpha = val[f]

	elif phi == 3:
		x, y = gaussianKernel(X)
		handle = 'Gaussian kernel'
		val = {'sph.txt': 5e-5, 'lin.txt': 1e-3}
		alpha = val[f]

	q, MSE, theta = gradient_descent(x, y, alpha, max_itr, initial_del)
	plot(q, MSE, title, handle, 'Percentage of data used(%)', 'Mean square error')
	y_predicted = x.dot(theta)
	x = X[:,0]
	scatter_plot(x, y, y_predicted, 'Training data and predicted data')
	return theta
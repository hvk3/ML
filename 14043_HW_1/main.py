from sklearn import metrics, manifold
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from my_kmeans import *

def plotGraph(points, labels, title):
	model = manifold.TSNE(learning_rate = 250)
	tsne_data = model.fit_transform(points)
	plt.scatter(tsne_data[:,0], tsne_data[:,1], c = labels)
	plt.title(title)
	plt.show()

print "Enter dataset name:",
filename = raw_input()
print "Enter K:",
K = int(raw_input())
temp = filename
title = temp.split('.')
	
X = loadData(filename)
Y = deepcopy(X)

groundTruth = []
for n in range(len(X)):
	groundTruth.append(X[n][-1])

plotGraph(X, groundTruth, filename + " : Ground truth scatter")
initial_centroids = generateCentroids(X, K)
arr1 = kMeans(X, initial_centroids, 20)
arr2 = GMM(X, K)
sum1 = np.zeros(len(arr1[1]))
sum2 = np.zeros(len(arr2[1]))

for j in range(10):
	print j + 1, "iteration(s) done."
	initial_centroids = generateCentroids(X, K)
	arr1 = kMeans(X, initial_centroids, 20)
	print "Metric values (MI, AMI, RI, ARI) for k-means:"
	for i in arr1[1]:
		print i
	sum1 += arr1[1]

for j in range(10):
	print j + 1, "iteration(s) done."
	arr2 = GMM(X, K)
	print "Metric values (MI, AMI, RI, ARI) for GMM:"
	for i in arr2[1]:
		print i
	sum2 += arr2[1]

print "Average value for k-means:", sum1 / 10
print "Average value for GMM:", sum2 / 10
	
for i in range(len(Y)):
	Y[i] = Y[i][:-1]

prediction = []
for n in range(len(Y)):
	pos = 0
	for k in range(len(arr1[0])):
		if norm(arr1[0][k], Y[n]) < norm(arr1[0][pos], Y[n]):
			pos = k
	prediction.append(pos)
plotGraph(X, prediction, filename + " : Clustering output scatter")
plotGraph(X, arr2[0], filename + " : GMM scatter")
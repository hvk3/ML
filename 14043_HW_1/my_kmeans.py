import numpy as np
import random
import re
import matplotlib.pyplot as plt
from sklearn import metrics, mixture
from copy import deepcopy

def rand_score(a, b):
	same = 0
	diff = 0
	N = len(a)
	for i in range(N):
		for j in range(i + 1, N):
			pair = (a[i], a[j])
			if a[i] == a[j]:	# In same cluster in ground truth
				if b[i] == b[j]:
					same += 1
			else:	# In different clusters in ground truth
				if b[i] != b[j]:
					diff += 1
	return ((same + diff) * 2.0) / (N * (N - 1))

def norm(a, b):
	return ((a - b) ** 2).sum()

def loadData(filename):
	file = open(filename, 'rb')
	X = []
	groundTruth = []
	clusters = {}
	count = 0
	for line in file:
		if line == "\n" or line[0] == ';' or line[0] == 'R':
			continue
		parsedLine = line.lstrip().rstrip().replace(' ',',').replace('\t',',').split(',')
		parsedLine = filter(None, parsedLine)
		if re.match("^\d+?\.\d+?$", parsedLine[0]) is None and parsedLine[0].isdigit() == False:
			label = parsedLine[0]
			parsedLine = parsedLine[1:]
			parsedLine.append(label)
		points = map(float, parsedLine[:-1])
		X.append(points)
		if parsedLine[-1] not in clusters:
			clusters[parsedLine[-1]] = count
			count += 1
		groundTruth.append(parsedLine[-1])
	for i in range(len(groundTruth)):
		for key in clusters:
			if groundTruth[i] == key:
				groundTruth[i] = clusters[key]
		X[i].append(groundTruth[i])
	return X

def generateCentroids(X, K):
	centroids = []
	for i in range(K):
		centroids.append(X[random.randint(1, 100000) % len(X)])
	return centroids

def kMeans(X, initial_centroids, max_iterations):

	N = len(X)
	K = len(initial_centroids)
	features = len(X[0]) - 1
	Y = deepcopy(X)

	groundTruth = []
	for n in range(N):
		groundTruth.append(Y[n][-1])

	for n in range(N):
		Y[n] = Y[n][:-1]

	for k in range(K):
		initial_centroids[k] = initial_centroids[k][:-1]
	
	points = np.array(Y)
	newCentroid = initial_centroids
	iterations = 0
	cost = []
	iternum = []

	while iterations != max_iterations:
		r = np.zeros((N, K))
		J = 0
		for n in range(N):
			pos = 0
			for k in range(K):
				if norm(points[n], newCentroid[pos]) > norm(points[n], newCentroid[k]):
					pos = k
			r[n][pos] = 1
		for k in range(K):
			new_centroid_numerator = np.zeros(features)
			new_centroid_denominator = 0
			for n in range(N):
				new_centroid_denominator += r[n][k]
				new_centroid_numerator += r[n][k] * points[n]
			if new_centroid_denominator != 0:
				newCentroid[k] = new_centroid_numerator / new_centroid_denominator
			else:
				continue
		for n in range(N):
			for k in range(K):
				J += norm(newCentroid[k], points[n]) * r[n][k]
		iterations += 1
		cost.append(J)
		iternum.append(iterations)

	# plt.plot(iternum, cost)
	# plt.title('Distortion per iteration')
	# plt.show()

	predictedClusters = []
	for n in range(N):
		for k in range(K):
			if r[n][k] == 1:
				predictedClusters.append(k)

	ARI = metrics.adjusted_rand_score(groundTruth, predictedClusters)
	RI = rand_score(groundTruth, predictedClusters)
	AMI = metrics.adjusted_mutual_info_score(groundTruth, predictedClusters)
	MI = metrics.normalized_mutual_info_score(groundTruth, predictedClusters)
	evaluationMatrix = [MI, AMI, RI, ARI]
	return newCentroid, evaluationMatrix

def GMM(X, K):
	g = mixture.GMM(n_components = K)
	Y = deepcopy(X)
	N = len(X)
	groundTruth = []
	for n in range(N):
		groundTruth.append(Y[n][-1])
	for n in range(N):
		Y[n] = Y[n][:-1]
	predictedClusters = g.fit_predict(Y)
	ARI = metrics.adjusted_rand_score(groundTruth, predictedClusters)
	RI = rand_score(groundTruth, predictedClusters)
	AMI = metrics.adjusted_mutual_info_score(groundTruth, predictedClusters)
	MI = metrics.normalized_mutual_info_score(groundTruth, predictedClusters)
	evaluationMatrix = [MI, AMI, RI, ARI]
	return predictedClusters, evaluationMatrix
from my_svm import *

import numpy as np
import matplotlib.pyplot as plt
import warnings

from math import log10 
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, normalize
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

trn_images, trn_labels = generateDataset(training_files[0], training_files[1], 2000)
tst_images, tst_labels = generateDataset(test_files[0], test_files[1], 500)

f1, f2, t1, t2 = [], [], [], []

normalize(trn_images)
normalize(tst_images)

def generateROC(scoreMatrix, trueLabels, X_label, y_label, nROCpts = 100, plotROC = False):
	tpr = np.zeros([1, nROCpts])
	fpr = np.zeros([1, nROCpts])

	nTrueLabels = np.count_nonzero(trueLabels) 
	nFalseLabels = np.size(trueLabels) - nTrueLabels

	minScore = np.min(scoreMatrix)
	maxScore = np.max(scoreMatrix);
	rangeScore = maxScore - minScore;

	thdArr = minScore + rangeScore*np.arange(0, 1, 1.0 / (nROCpts))
	# print thdArr
	for thd_i in range(0,nROCpts):
		thd = thdArr[thd_i]
		ind = np.where(scoreMatrix >= thd) 
		thisLabel = np.zeros([np.size(scoreMatrix, 0), np.size(scoreMatrix, 1)])
		thisLabel[ind] = 1
		tpr_mat = np.multiply(thisLabel, trueLabels)
		tpr[0, thd_i] = np.sum(tpr_mat) / nTrueLabels
		fpr_mat = np.multiply(thisLabel, 1 - trueLabels)
		fpr[0, thd_i] = np.sum(fpr_mat) / nFalseLabels

		# print fpr
		# print tpr
	if plotROC:
		plt.xlim(min(fpr[0,:]), max(fpr[0,:]))
		plt.plot(fpr[0,:], tpr[0,:], 'b.-')
		plt.xlabel(X_label)
		plt.ylabel(y_label)
		plt.show()

	return fpr,tpr,thdArr

def plotGraph(X, y, X_label, y_label):
	plt.plot(X, y)
	plt.xlabel(X_label)
	plt.ylabel(y_label)
	plt.show()

def plotGraphs(X1, y1, X2, y2, X_label, y_label):
	plt.plot(X1, y1, color = 'blue', label = 'Model for Linear kernel')
	plt.plot(X2, y2, color = 'red', label = 'Model for RBF kernel')
	plt.scatter(X1, y1, color = 'blue', label = 'Model for Linear kernel')
	plt.scatter(X2, y2, color = 'red', label = 'Model for RBF kernel')
	plt.xlabel(X_label)
	plt.ylabel(y_label)
	plt.show()

def train_and_classify(X_train, y_train, X_test, y_test, C, K = 5, kernel = ['linear'], iterations = [3000], gamma = ['auto'], verbose = False):
	distinct_labels = dict()
	for label in y_train:
		if label not in distinct_labels:
			distinct_labels[label] = 1
	classes = np.array(distinct_labels.keys())

	binarized_y_train = label_binarize(y_train, classes = classes)
	binarized_y_test = label_binarize(y_test, classes = classes)

	param_grid = {'C' : C, 'kernel' : kernel, 'gamma' : gamma, 'max_iter' : iterations}
	classifier = SVC()
	kfoldclassifier = GridSearchCV(classifier, param_grid, cv = K, verbose = verbose, n_jobs = 8)
	kfoldclassifier.fit(X_train, y_train)
	c_vs_errors = kfoldclassifier.cv_results_['mean_test_score']
	best_classifier = kfoldclassifier.best_estimator_
	C = best_classifier.C
	gamma = best_classifier.gamma
	ovr = OneVsRestClassifier(best_classifier, n_jobs = -1)
	ovr.fit(X_train, binarized_y_train)

	return ovr, ovr.score(X_test, binarized_y_test) * 100, C, gamma, c_vs_errors

if __name__ == '__main__':
	while True:
		print '1. Linear SVM for binary classification.'
		print '2. Linear SVM for multiclass classification.'
		print '3. RBF SVM for multiclass classification.'
		print '4. Exit'
		print 'Enter option:',
		x = int(raw_input())
		if x > 4:
			print 'Invalid option entered. Please try again.'
			continue
		elif x == 4:
			break
		else:
			if x == 1:
				print 'Enter first digit/class:',
				a = int(raw_input())
				print 'Enter second digit/class:',
				b = int(raw_input())

				print 'Processing data...'

				C = [1e-6, 1e-3, 1e0, 1e3, 1e6]
				binary_trn_images, binary_trn_labels = binaryClassDataset(trn_images, trn_labels, a, b)
				binary_tst_images, binary_tst_labels = binaryClassDataset(tst_images, tst_labels, a, b)
				classifier, accuracy, C1, gamma1, c_vs_errors = train_and_classify(binary_trn_images, binary_trn_labels, binary_tst_images, binary_tst_labels, C, verbose = True)
				print 'Accuracy of binary classifier with classes '  + str(a) + ' and ' + str(b) + ' : ' + str(accuracy) + '%'

				for i in range(len(c_vs_errors)):
					c_vs_errors[i] = (1 - c_vs_errors[i]) * 100
					C[i] = log10(C[i])
				plotGraph(C, c_vs_errors, 'log10C', 'Cross-validation error(%)')
				t1 = classifier.decision_function(binary_tst_images)
				t2 = np.zeros((len(np.unique(tst_labels)), tst_labels.shape[0]))
				for i in range(len(np.unique(tst_labels))):
					t2[i, :] = (tst_labels == i)
				t2 = (binary_tst_labels == a) * 1
				fpr, tpr, thdArr = generateROC(t1.transpose(), t2, nROCpts = t2.shape[0], plotROC = True, X_label = 'False positive rate', y_label = 'Classification accuracy(True positive rate)')
				joblib.dump(classifier, '../Models/model_linear.model')
			elif x == 2:
				print 'Processing data...'
				C = [1e-7, 1e-3, 1e1, 1e5]
				classifier, accuracy, C1, gamma1, c_vs_errors = train_and_classify(trn_images, trn_labels, tst_images, tst_labels, C, verbose = True)
				print 'Accuracy of linear kernel multiclassifier: ' + str(accuracy) + '%'
				for i in range(len(c_vs_errors)):
					c_vs_errors[i] = (1 - c_vs_errors[i]) * 100
					C[i] = log10(C[i])
				plotGraph(C, c_vs_errors, 'log10C', 'Cross-validation error(%)')
				t1 = classifier.decision_function(tst_images)
				t2 = np.zeros((len(np.unique(tst_labels)), tst_labels.shape[0]))
				for i in range(len(np.unique(tst_labels))):
					t2[i, :] = (tst_labels == i)
				fpr, tpr, thdArr = generateROC(t1.transpose(), t2, nROCpts = t2.shape[0], plotROC = True, X_label = 'False positive rate', y_label = 'Classification accuracy(True positive rate)')
				f1, t1 = fpr, tpr
				for i in range(len(classifier.classes_)):
					joblib.dump(classifier.estimators_[i], '../Models/multi' + str(i) + '.model')
			else:
				print 'Processing data...'
				C = [1e-7, 1e2]
				gamma = [1e-6, 10]
				classifier, accuracy, C1, gamma1, c_vs_errors = train_and_classify(trn_images, trn_labels, tst_images, tst_labels, C, gamma = gamma, kernel = ['rbf'], verbose = True)
				print 'Accuracy of RBF kernel multiclassifier: ' + str(accuracy) + '%'
				
				# for i in range(len(c_vs_errors)):
				# 	c_vs_errors[i] = (1 - c_vs_errors[i]) * 100
				# for i in range(len(C)):
				# 	C[i] = log10(C[i])
				# for i in range(len(gamma)):
				# 	gamma[i] = log10(gamma[i])
				
				# plotGraph(C, c_vs_errors, 'log10C', 'Cross-validation error(%)')
				# plotGraph(gamma, c_vs_errors, 'log10gamma', 'Cross-validation error(%)')

				# t1 = classifier.decision_function(tst_images)
				# t2 = np.zeros((len(np.unique(tst_labels)), tst_labels.shape[0]))
				# for i in range(len(np.unique(tst_labels))):
				# 	t2[i, :] = (tst_labels == i)
				# fpr, tpr, thdArr = generateROC(t1.transpose(), t2, nROCpts = t2.shape[0], plotROC = True, X_label = 'False positive rate', y_label = 'Classification accuracy(True positive rate)')
				# f2, t2 = fpr, tpr
				# if f1 != [] and f2 != []:
				# 	plotGraphs(f1, t1, f2, t2, X_label = 'False positive rate', y_label = 'Classification accuracy(True positive rate)')
				for i in range(len(classifier.classes_)):
					joblib.dump(classifier.estimators_[i], '../Models/rbf' + str(i) + '.model')
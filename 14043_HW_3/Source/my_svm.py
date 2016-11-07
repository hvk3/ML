import numpy as np
import matplotlib.pyplot as plt
import struct
from sklearn.utils import shuffle

'''
File parser source code from:
https://docs.python.org/2/library/struct.html
https://gist.github.com/akesling/5358964
'''

training_files = ['../train-images.idx3-ubyte', '../train-labels.idx1-ubyte']
test_files = ['../t10k-images.idx3-ubyte', '../t10k-labels.idx1-ubyte']

def parseLabels(fileName):
	f = open(fileName, 'rb')
	magic, size = struct.unpack('>II', f.read(8))
	labels = np.fromfile(f, dtype = np.uint8)
	return labels

def parseImages(fileName):
	f = open(fileName, 'rb')
	magic, size, rows, columns = struct.unpack('>IIII', f.read(16))
	images = np.fromfile(f, dtype = np.uint8).reshape(size, rows, columns)
	return images

def generateDataset(img_fileName, lbl_fileName, images_per_label):
	images, labels = parseImages(img_fileName), parseLabels(lbl_fileName)
	images, labels = shuffle(images, labels)
	num_labels = len(np.unique(labels))
	dataset_images = np.zeros([images_per_label * num_labels, images[0].shape[0] * images[0].shape[1]])
	dataset_labels = np.zeros(images_per_label * num_labels)
	dataset_images_per_label = np.zeros(num_labels)

	j = 0
	for i in range(len(images)):
		label = labels[i]
		if dataset_images_per_label[label] == images_per_label:
			continue
		dataset_images_per_label[label] += 1
		dataset_images[j] = images[i].reshape(images[0].shape[0] * images[0].shape[1])
		dataset_labels[j] = label
		j += 1
	dataset_images, dataset_labels = shuffle(dataset_images, dataset_labels)
	return dataset_images, dataset_labels

def binaryClassDataset(images, labels, x, y):
	num_labels = 0
	for i in range(len(labels)):
		if labels[i] == x:
			num_labels += 1
	
	res_images = np.zeros([num_labels * 2, images[0].shape[0]])
	res_labels = np.zeros(num_labels * 2)
	
	j, k = 0, 0
	for i in range(len(labels)):
		if labels[i] == x or labels[i] == y:
			res_images[j] = images[i]
			res_labels[j] = labels[i]
			j += 1
	res_images, res_labels = shuffle(res_images, res_labels)
	return res_images, res_labels
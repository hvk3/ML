import numpy as np
import matplotlib.pyplot as plt
import itertools

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import Adadelta
from keras.models import load_model

from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

def plot_confusion_matrix(cm, classes, normalize=False, title = 'Confusion matrix', cmap = plt.cm.Blues):
	plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

model = Sequential()

(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train, y_train) = shuffle(X_train, y_train)
(X_test, y_test) = shuffle(X_test, y_test)

size = X_train.shape[1] * X_train.shape[2]
X_train, X_test = X_train.reshape(X_train.shape[0], size).astype('float32'), X_test.reshape(X_test.shape[0], size).astype('float32')

X_train /= 255.0
X_test /= 255.0

binarized_y_train, binarized_y_test = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test)

model.add(Dense(input_dim = size, output_dim = 500, activation = 'tanh'))
model.add(Dense(input_dim = 500, output_dim = 250, activation = 'tanh'))
model.add(Dense(input_dim = 250, output_dim = binarized_y_test.shape[1], activation = 'softmax'))

learning_rates = [0.1, 0.3, 0.5, 0.7]

for lr in learning_rates:
	model.compile(loss = 'categorical_crossentropy', optimizer = Adadelta(lr = lr), metrics = ['accuracy'])
	model.fit(X_train, binarized_y_train, nb_epoch = 20, batch_size = 256, verbose = 0)
	scores = model.evaluate(X_test, binarized_y_test, verbose = 0)
	print 'Test accuracy for learning rate ' + str(lr) + ': ' + str(scores[1] * 100) + '%'

	binarized_y_pred_temp = model.predict(X_test)
	y_pred = np.zeros(binarized_y_test.shape[0])
	
	for i in range(len(binarized_y_pred_temp)):
		for j in range(len(binarized_y_pred_temp[i])):
			binarized_y_pred_temp[i][j] = round(binarized_y_pred_temp[i][j])
			if binarized_y_pred_temp[i][j] == 1:
				y_pred[i] = j
	cm = confusion_matrix(y_test, y_pred)
	plot_confusion_matrix(cm, classes = np.unique(y_train), title = 'Confusion matrix for learning rate ' + str(lr))
	plt.show()
	model.save('./FFNN/FFNN_' + str(lr) + '.h5')
	# model = load_model('./FFNN/FFNN_' + str(lr) + '.h5')
	'''
	Test accuracy for learning rate 0.1: 94.9%
	Test accuracy for learning rate 0.3: 97.44%
	Test accuracy for learning rate 0.5: 97.97%
	Test accuracy for learning rate 0.7: 98.22%
	'''
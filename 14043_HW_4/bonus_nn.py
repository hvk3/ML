# Reference for autoencoders : https://blog.keras.io/building-autoencoders-in-keras.html

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adadelta
from keras.datasets import mnist
from keras.utils import np_utils

from sklearn.utils import shuffle

model = Sequential()

(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train, y_train) = shuffle(X_train, y_train)
(X_test, y_test) = shuffle(X_test, y_test)

size = X_train.shape[1] * X_train.shape[2]
X_train, X_test = X_train.reshape(X_train.shape[0], size).astype('float32'), X_test.reshape(X_test.shape[0], size).astype('float32')

X_train /= 255.0
X_test /= 255.0

binarized_y_train, binarized_y_test = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test)

learning_rates = [0.5, 0.7, 0.9, 1.1, 1.3]
errors = np.zeros(5)
k = 0

img_input = Input(shape = (size,))
encoded_input = Dense(100, activation = 'sigmoid')(img_input)
decoded_input = Dense(size, activation = 'sigmoid')(encoded_input)

encoder = Model(input = img_input, output = encoded_input)
autoencoder = Model(input = img_input, output = decoded_input)

autoencoder.compile(loss = 'binary_crossentropy', optimizer = 'Adadelta')
autoencoder.fit(X_train, X_train, nb_epoch = 20, batch_size = 128, verbose = 0)

compressed_X_train = encoder.predict(X_train)
compressed_X_test = encoder.predict(X_test)

model.add(Dense(input_dim = 100, output_dim = 50, activation = 'sigmoid'))
model.add(Dense(input_dim = 50, output_dim = 10, activation = 'softmax'))

for lr in learning_rates:
	model.compile(loss = 'categorical_crossentropy', optimizer = Adadelta(lr = lr), metrics = ['accuracy'])
	model.fit(compressed_X_train, binarized_y_train, nb_epoch = 50, batch_size = 128, verbose = 0)

	scores = model.evaluate(compressed_X_test, binarized_y_test, verbose = 2)

	print 'Test accuracy for learning rate ' + str(lr) + ': ' + str(scores[1] * 100) + '%'
	errors[k] = 100 - scores[1] * 100
	k += 1

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(learning_rates, errors)
plt.title('Learning rates vs Classification errors(%)')
plt.xlabel('Learning rates')
plt.ylabel('Classification errors(%)')

for xy in zip(learning_rates, errors):
	ax.annotate('(%s, %s)' % xy, xy = xy, textcoords = 'data')
plt.show()
'''
Test accuracy for learning rate 0.5: 85.78%
Test accuracy for learning rate 0.7: 90.13%
Test accuracy for learning rate 0.9: 91.81%
Test accuracy for learning rate 1.1: 92.58%
Test accuracy for learning rate 1.3: 93.09%
'''
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt


seed = 7
np.random.seed(seed)


(x_train, y_train), (x_test, y_test) = mnist.load_data()


plt.subplot(221)
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()


# initialize parameters
learning_rate = 0.00001
epochs = 10
batch_size = 200
max_train_items = x_train.shape[0]
max_test_items = x_test.shape[0]

X_train = x_train[:max_train_items]
Y_train = y_train[:max_train_items]
X_test = x_test[:max_test_items]
Y_test = y_test[:max_test_items]


# Flatten the input values
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Normalize inputs from 0-1
X_train = X_train/255
X_test = X_test/255


# One hot encode output
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

num_classes = Y_train.shape[1]


# define a simple multilayer perceptron
model = Sequential();
model.add(Dense(300, input_dim=num_pixels));
model.add(Activation('relu'));
model.add(Dense(100));
model.add(Activation('relu'));
model.add(Dense(num_classes));
model.add(Activation('softmax'));

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def initializeTFModel(learning_rate):
	#defining a 2D array of two float variables
	x = tf.placeholder(tf.float32, [None, 2])

	#define weights of dim 2x2
	W = tf.Variable(tf.zeros([2,2]))

	#define biases - array of two
	b = tf.Variable(tf.zeros([2]))

	#define the first layer
	y_values = tf.add(tf.matmul(x, W), b)

	#define the activation function
	y = tf.nn.softmax(y_values)

	#define a placeholder for predicted outputs
	y_ = tf.placeholder(tf.float32, [None, 2])

	#define the cost function
	cost = tf.reduce_sum(tf.pow(y - y_, 2)/2.)

	#define the optimizer - gradient descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	return x, y_, W, b, cost, optimizer

def captureData():
	#capture the input data
	dataframe = pd.read_csv("house_data.csv")
	dataframe = dataframe.drop(["index", "price", "sq_price"], axis=1)
	#use only the first 10 entries
	dataframe = dataframe[:10]

	#set output features
	dataframe.loc[:, ("y1")] = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
	dataframe.loc[:, ("y2")] = dataframe["y1"] == 0  
	dataframe.loc[:, ("y2")] = dataframe["y2"].astype(int)

	#Initialize the input and output params
	inputX = dataframe.loc[:, ['area', 'bathrooms']].as_matrix()
	inputY = dataframe.loc[:, ["y1", "y2"]].as_matrix()
	
	return inputX, inputY

def run():
	inputX, inputY = captureData()

	#Initialize paramters
	learning_rate = 0.000001
	training_epochs = 2000
	display_step = 50
	n_samples = inputY.size

	#tensflow initialize
	x, y_, W, b, cost, optimizer = initializeTFModel(learning_rate)
	cost = cost/ n_samples

	#Initialize variabls and tensorflow session
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)

	#Training the model
	for i in range(training_epochs):  
		# Take a gradient descent step using our inputs and labels
	    sess.run(optimizer, feed_dict={x: inputX, y_: inputY}) 

	    # Display logs per epoch step
	    if (i) % display_step == 0:
	        cc = sess.run(cost, feed_dict={x: inputX, y_:inputY})
	        print "Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc)

	print "Optimization Finished!"
	training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
	print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n' 


if __name__ == '__main__':
	run()
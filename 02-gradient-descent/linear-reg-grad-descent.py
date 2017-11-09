from numpy import *
import matplotlib.pyplot as plt

def computeError(points, m, b):
	totalError = 0
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m*x + b)) ** 2
	return totalError/float(len(points))

def stepGradient(points, m, b, learning_rate):
	gradient_m = 0
	gradient_b = 0
	N = float(len(points))
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		gradient_m += -(2/N) * x * (y - ((m * x) + b))
		gradient_b += -(2/N) * (y - ((m * x) + b))
	m = m - gradient_m * learning_rate
	b = b - gradient_b * learning_rate
	return [m ,b]


def gradientDescentRunner(points, initial_m, initial_b, learning_rate, num_iters):
	m = initial_m
	b = initial_b
	err = []

	for i in range(num_iters):
		m ,b = stepGradient(array(points), m , b, learning_rate)
		err.append(computeError(points, m, b))
		print "Gradient descent run #" + str(i) + ": m = " + str(m) + " b = " + str(b) + " err = " + str(err[-1]) 
	return [m, b, err]

def run():
	points = genfromtxt("dataset-grad-des.csv", delimiter=",")
	initial_b = 0
	initial_m = 0
	learning_rate = 0.0002
	num_iters = 500

	print "Starting the gradient decent for linear regression with m = {0}, b = {1}, error = {2}".format(initial_m, initial_b, computeError(points, initial_m, initial_b))
	print "Running with learning rate of " + str(learning_rate) + " ... "
	[m, b, err] = gradientDescentRunner(points, initial_m, initial_b, learning_rate, num_iters)
	print "Gradient decent ran successfully and fit with m = {0}, b = {1} and error = {2}".format(m, b, err[-1])

	# Prediction plots
	plt.scatter(points[:,0], points[:, 1], color='red')
	plt.plot(points[:,0], [m * i + b for i in points[:,0]], color='blue')
	plt.show()

	# Error plots
	plt.plot(range(len(err)), err)
	plt.show()


if __name__ == '__main__':
	run()

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
	def __init__(self, num_attr, num_neurons):
		# seed the random generator to give same numbers every run
		np.random.seed(1)

		# Modelling a single neuron with 3 input connections
		# and 1 output connection

		# Weights will be assigned to a 3x1 matrix with values [-1, 1] mean 0
		self.synaptic_weights = 2 * np.random.random((num_attr, num_neurons)) - 1

		# error logs
		self.mean_error_log = []

	# Activation function to normalize the results to [-1, 1]
	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	# Derivative of the activation function for weights adjustments
	def __sigmoid_derivative(self, x):
		return x * (1 -x)

	def train(self, training_input, training_output, num_iters):
		for iteration in xrange(num_iters):
			# Pass the training set through our neural network
			model_output = self.think(training_input)

			# Calculate the error 
			error = training_output - model_output
			
			# Gradient descent for weight adjustments
			adj = np.dot(training_input.T, self.__sigmoid_derivative(model_output) * error)

			# Adjust the weights
			self.synaptic_weights += adj

			# print "Epoch #{0}: Error = {1}".format(iteration+1, np.mean(error))
			
			self.mean_error_log.append(np.mean(error))


	# Output processing
	def think(self, training_input):
		# Pass inputs through our neural network
		return self.__sigmoid(np.dot(training_input, self.synaptic_weights))

	# Plot the error logs
	def plot_error(self):
		print "Trained Weights: " + str(self.synaptic_weights)
		plt.plot(range(len(self.mean_error_log)), self.mean_error_log, color='blue')
		plt.show()


def run():
	# training data with input and output - price = func(rooms, baths, floors)
	points = np.genfromtxt('zoo_data.csv', delimiter=",", dtype=float)
	# points = np.array([[0, 0, 1, 0], [1, 1, 1, 1], [1, 0, 1, 1], [0, 1, 1, 0]])

	# Set basic [params
	num_iters = 5000
	num_attr = 14
	num_neurons = 1

	print "Loaded zoo dataset for animal type classification"

	# Initialize the neural network
	neural_network = NeuralNetwork(num_attr, num_neurons)

	# Random initializing the synaptic weights
	print "Initializing the synaptic weights"
	print neural_network.synaptic_weights

	# Traing the neural network
	neural_network.train(points[:,:num_attr], points[:,num_attr:num_attr+1], num_iters)

	# Plot error
	neural_network.plot_error()


if __name__ == "__main__":
	run()
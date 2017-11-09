import numpy as np

class NeuralNetwork:
	def __init__(self):
		# seed the random generator to give same numbers every run
		np.random.seed(1)

		# Modelling a single neuron with 3 input connections
		# and 1 output connection

		# Weights will be assigned to a 3x1 matrix with values [-1, 1] mean 0
		self.synaptic_weights = 2 * np.random.random((3,1)) - 1

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
			error = training_output - model_output.T

			# Gradient descent for weight adjustments
			adj = np.dot(training_input.T, self.__sigmoid_derivative(model_output) * error.T)

			# Adjust the weights
			self.synaptic_weights += adj

			print "Epoch #{0}: Error = {1}".format(iteration, sum(error))

	# Output processing
	def think(self, training_input):
		# Pass inputs through our neural network
		return self.__sigmoid(np.dot(training_input, self.synaptic_weights))


def run():
	# training data with input and output - price = func(rooms, baths, floors)
	points = np.genfromtxt('kc_house_data.csv', delimiter=",", dtype=float)
	num_iters = 10

	print "Loaded house price correlation dataset"

	# Initialize the neural network
	neural_network = NeuralNetwork()

	# Random initializing the synaptic weights
	print neural_network.synaptic_weights

	# Traing the neural network
	neural_network.train(points[:,1:], points[:,0], num_iters)


if __name__ == "__main__":
	run()
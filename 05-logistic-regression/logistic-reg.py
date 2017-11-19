import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt


# Lifecyle maintenance function
def run():
	# constants
	num_features = 4

	# capture the data
	data = pd.read_csv('iris_dataset.csv', header=None)
	data = data[1:]
	x_values = data[range(num_features)]
	y_values = data[4]

	# preprocess the data
	LE = preprocessing.LabelEncoder()
	LE.fit(y_values)
	y_values = LE.transform(y_values)
	print "Classes from input data : " + LE.classes_

	# train the data
	model = linear_model.LogisticRegression()
	model.fit(x_values, y_values)

	# predict
	y_predict = model.predict(x_values)
	accuracy = metrics.accuracy_score(y_values, y_predict)
	print "Accuracy Score = {0}%".format(accuracy*100)

	# visualize the results
	plt.scatter(x_values[0], y_values, color='red')
	plt.scatter(x_values[0], y_predict, color='blue')
	plt.show()

	# convert result to classes
	y_values = LE.inverse_transform(y_values)


# Main function
if __name__ == '__main__':
	run()
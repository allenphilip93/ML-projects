import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt

# import the dataset
dataframe = pd.read_csv('dataset-lin-reg.txt', header=None)
x_values = dataframe[0].reshape(len(dataframe[0]), 1)
y_values = dataframe[1]

# train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# evaluate the error
y_predict = body_reg.predict(x_values)
y_err = sqrt(mean_squared_error(y_predict, y_values))
print "RMS Error of Model: " + str(y_err)

# visualize the results
plt.scatter(x_values, y_values, color='red')
plt.plot(x_values, body_reg.predict(x_values), color='blue', linewidth=3)
plt.show()
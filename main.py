import numpy as np
import matplotlib.pyplot as plt
import Regression as reg

# Load CSV File
data = np.genfromtxt('framingham.csv', delimiter=',',usecols=(0,5,1,4,7,8,9,10,11,14,15), skip_header=True)


# Cleaning the data
mean = np.nanmean(data, axis=0)	# Calculate mean of each column
nan_indices = np.where(np.isnan(data))
data[nan_indices] = np.take(mean, nan_indices[1])	# Replace nan with mean

# Normalization except the last column
for i in range(data.shape[1] - 1):
	data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])


# Set the seed
np.random.seed(0)
# Splitting the data 40% training, 40% cross validation, 20% test
np.random.shuffle(data)
train_data = data[:int(0.4*len(data))]
cv_data = data[int(0.4*len(data)):int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]
print("Training data: ", train_data.shape)
print("Cross Validation data: ", cv_data.shape)
print("Test data: ", test_data.shape)

# Training the model - y is the last column
X = train_data[:, :-1]
y = train_data[:, -1]
# Reshape y to be a column vector
y = y.reshape(y.shape[0], 1)
print("Features shape: ", X.shape)
print("Labels shape: ", y.shape)
# theta with zeros
theta = np.zeros((X.shape[1], 1))
theta = reg.linear_regression(X, y, theta, 0.05, 0.002)
print("Theta shape: ", theta.shape)

# Testing the model
X = test_data[:, :-1]
y = test_data[:, -1]
y = y.reshape(y.shape[0], 1)

# Predicting the labels
y_pred = reg.sigmoid(theta, X)
y_pred = np.where(y_pred > 0.5, 1, 0)
print(y_pred)

# Get the accuracy of the model
accuracy = np.sum(y_pred == y) / len(y)
print("Accuracy: ", accuracy)
# Get the presicion of the model
precision = np.sum(y_pred[y == 1] == 1) / np.sum(y_pred == 1)
print("Precision: ", precision)





# https://www.kaggle.com/code/adithyabshetty100/coronary-heart-disease-prediction/notebook#Feature-Selection
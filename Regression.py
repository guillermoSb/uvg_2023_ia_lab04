import numpy as np
import matplotlib.pyplot as plt
# 0. Create logistic function
def sigmoid(theta, X):
		return 1/(1 + np.exp(X@-theta))


# 1. Create loss function
def loss(X, y, theta):
		N = len(y)
		return 1/(2*N) * np.sum((sigmoid(theta, X) - y)**2)

# 2. Create gradient function
def gradient(X, y, theta):
		N = len(y)
		return 1/N * X.T@((sigmoid(theta, X) - y) * sigmoid(theta, X) * (1 - sigmoid(theta, X)))

# 3. Create linear regression function
def linear_regression(X, y, theta, learning_rate, threshold = 0.05):
		# Store the squad error per example
		errors = []
		while True:
			theta = theta - learning_rate * gradient(X, y, theta)
			# Append the squad error per example
			errors.append(loss(X, y, theta))
			if np.sqrt(np.sum(gradient(X, y, theta)**2)) < threshold:
				# Plot the errors
				plt.plot(errors)
				plt.xlabel("Iterations")
				plt.ylabel("Loss")
				plt.show()

				return theta
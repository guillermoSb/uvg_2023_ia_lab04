import numpy as np
import matplotlib.pyplot as plt
import Regression as reg

# Load CSV File
data = np.genfromtxt('data.csv', delimiter=',',usecols=(0,1,4,5,7,8,9,10,11,14,15), skip_header=True)


# Cleaning the data
mean = np.nanmean(data, axis=0)	# Calculate mean of each column
nan_indices = np.where(np.isnan(data))
data[nan_indices] = 0
# Normalization except the last column
for i in range(data.shape[1] - 1):
	data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])

ages = data[:,1]
cigs = data[:,2]
data = np.hstack((np.ones((data.shape[0], 1)), data))	# 4
data = np.hstack((ages.reshape(data.shape[0], 1), data))	# 3
data = np.hstack((ages.reshape(data.shape[0], 1), data))	# 1
data = np.hstack((cigs.reshape(data.shape[0], 1), data))	# 1
data = np.hstack((cigs.reshape(data.shape[0], 1), data))	# 0
k = 1
# Add a polynomic feature to the data, third column with hstack

# Set the seed
np.random.seed(4)
# Splitting the data 40% training, 40% cross validation, 20% test
np.random.shuffle(data)
train_data = data[:int(0.4*len(data))]
cv_data = data[int(0.4*len(data)):int(0.8*len(data))]



print("Training data: ", train_data.shape)
print("Cross Validation data: ", cv_data.shape)

# theta with zeros


train_losses = []
cv_losses = []
theta = np.ones((data.shape[1], 1))
best_theta = None
for k in range(1,5):
	print(k)
	# Replace column 0 with the column 3 to the power of k
	train_data[:,0] = train_data[:,7] ** k-1
	train_data[:,1] = train_data[:,7] ** max(0, k-2)
	train_data[:,2] = train_data[:,6] ** max(0, k)
	train_data[:,3] = train_data[:,6] ** max(0, k-3)

	
	cv_data[:,0] = cv_data[:,7] ** k-1
	cv_data[:,1] = cv_data[:,7] ** max(0, k-2)
	cv_data[:,2] = cv_data[:,6] ** max(0, k)
	cv_data[:,3] = cv_data[:,6] ** max(0, k-3)
	

	X_train = train_data[:, :-1]
	y_train = train_data[:, -1]
	y_train = y_train.reshape(y_train.shape[0], 1)

	
	theta = np.ones((X_train.shape[1], 1))
	theta = reg.linear_regression(X_train, y_train, theta, 0.05, 0.05)
	# J for training data
	y_pred_train = reg.sigmoid(theta, X_train)
	y_pred_train = np.where(y_pred_train > 0.5, 1, 0)
	loss_train = reg.loss(X_train, y_train, theta)
	train_losses.append(loss_train)

	# J for cross validation data
	X_cv = cv_data[:, :-1]
	y_cv = cv_data[:, -1]
	y_cv = y_cv.reshape(y_cv.shape[0], 1)
	y_pred_cv = reg.sigmoid(theta, X_cv)
	y_pred_cv = np.where(y_pred_cv > 0.5, 1, 0)
	loss_cv = reg.loss(X_cv, y_cv, theta)
	cv_losses.append(loss_cv)
	if k == 3:
		best_theta = theta
print(train_losses)
plt.plot(train_losses, label='Training Loss')
plt.plot(cv_losses, label='Validation Loss')
plt.legend()
plt.xlabel("Iterations")
plt.show()


best_k = 3



test_data = data[int(0.8*len(data)):]

test_data[:,0] = test_data[:,7] ** best_k-1
test_data[:,1] = test_data[:,7] ** max(0, best_k-2)
test_data[:,2] = test_data[:,6] ** max(0, best_k)
test_data[:,3] = test_data[:,6] ** max(0, best_k-3)

# # J for test data
X_test = test_data[:, :-1]
y_test = test_data[:, -1]
y_test = y_test.reshape(y_test.shape[0], 1)

y_pred_test = reg.sigmoid(theta, X_test)
y_pred_test = np.where(y_pred_test > 0.5, 1, 0)
loss_test = reg.loss(X_test, y_test, theta)
accuracy_test = np.sum(y_pred_test == y_test) / len(y_test)
precision_test = np.sum(y_pred_test[y_test == 1] == 1) / np.sum(y_pred_test == 1)
print("Accuracy for test data: ", accuracy_test)
print("Precision for test data: ", precision_test)
# print("Loss for test data: ", loss_train)



# print(np.sum(y_pred[y == 1] == 1), np.sum(y_pred == 1))


# https://www.kaggle.com/code/adithyabshetty100/coronary-heart-disease-prediction/notebook#Feature-Selection
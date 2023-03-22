import numpy as np
import matplotlib.pyplot as plt

# Load CSV File
# Columns to use sysBP, glucose, TenYearCHD
data = np.genfromtxt('framingham.csv', delimiter=',',usecols=(10,14,15), skip_header=True)


# Cleaning the data
mean = np.nanmean(data, axis=0)	# Calculate mean of each column
nan_indices = np.where(np.isnan(data))
data[nan_indices] = np.take(mean, nan_indices[1])	# Replace nan with mean


# Plot sysBP vs glucose with TenYearCHD as color
plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap='bwr')
plt.xlabel('sysBP')
plt.ylabel('glucose')
plt.show()

# how cmap works
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

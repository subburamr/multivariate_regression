# ------------------------------------------------------------------------------------------------------------------------------------
# Title             : Multivariate Regression using Gradient Descent
# Description       : This is an implementation of gradient descent algorithm to minimize the least squares error for Multivariate Linear regression.
# Author            : Subburam Rajaram
# IMAT Number       : 03658122
# ------------------------------------------------------------------------------------------------------------------------------------

import numpy as np

# Initialize constants
stepsize = 0.01
iterations = 1000

# Read dataset from disk and structure it
Data = np.loadtxt("airfoil_self_noise.dat")
rows = Data.shape[0]
columns = Data.shape[1]
Y = Data[:,columns - 1 ]        # Labels
Y = Y.reshape((rows,1))
F = Data[:,0:(columns - 1)]     # Features

# Feature scaling and Mean normalization
F_normed = np.empty_like(F)
F_mean = F.mean(0)
F_std = F.std(0)
for i in range(0, rows):
    F_normed[i,:] = np.divide(( F[i,:] - F_mean ), F_std)

# stack column of ones to X for the constant term
ones = np.ones(rows)
X = np.column_stack((ones,F_normed))

# Weight Attribute Vector where W[0] would hold the value of constant term
Weight = np.zeros((columns, 1))
Cost = np.zeros((iterations, 1))

for iter in range(0, iterations):
    print "\nIteration:", iter
    Cost[iter] = (0.5/rows) * np.dot((np.dot(X, Weight) - Y).T, (np.dot(X, Weight) - Y))
    print "Cost:", Cost[iter]
    gradient = np.dot(X.T, (np.dot(X, Weight) - Y))/rows
    Weight = Weight - stepsize * gradient
    print "Intercept b: ", Weight[0]
    print "Weight W: ", Weight[1:columns].T

import numpy as np

### Initalize variables

stepsize = 0.5
iterations = 100

#will be using W0 as b

# W1 = np.empty_like(W)
# W1[0] = 132.83
# W1[1] = -0.0013
# W1[2] = -0.4221
# W1[3] = -35.6741
# W1[4] = 0.0999
# W1[5] = -147.2917

#W = W1
dJdW = np.zeros(3)
gradient = np.zeros(3)

### Read data set and structure it
#Data = np.loadtxt("airfoil_self_noise.dat")
Data = np.loadtxt("test")
columns = Data.shape[1]
Y = Data[:,columns - 1 ]
F = Data[:,0:(columns - 1)]
F_normed = (F - F.mean(0)) / (F.max(0) - F.min(0))
#F_normed = (F - F.mean(0)) / F.std(0)

ones = np.ones(Data.shape[0])
X = np.column_stack((ones,F_normed))
#print X


#gradient calculation
# def grad(X, Y, W):
#     #print "\nValue of dJdW is", dJdW
#     for j in range(len(W)):
#         sum = 0
#         for i in range(len(Y)):
#             sum+= ( np.dot(X[i,:],W) - Y[i] ) * X[i,j]
#          #   sum+= (Y[i] - (np.dot(W,X[i,:]))) * X[i,j]
#         print sum/len(Y)
#         dJdW[j] = sum/len(Y)
#     print "\nValue of dJdW is", dJdW
#     return dJdW



#grad descent implementation
def grad(X, Y, W):
    H = np.dot(X,W)
    error = (H - Y)
    gradient = np.dot(X.T, error) / len(Y)

 #   for i in range(len(W)):
 #       #temp = error * X[:,i]
 #       temp = np.dot(X[:,i].T, error)
 #       temp1[i] = W[i] - stepsize * (1.0 / len(Y)) * temp.sum()
    return gradient

def cost():
    H = X.dot(W)
    sqErrors = ( H - Y ) ** 2
    cost = (1.0 / (2 * len(Y))) * sqErrors.sum()
    #cost = 0
    #for i in range(len(Y)):
    #   cost += ( np.dot(X[i,:],W) - Y[i] ) * ( np.dot(X[i,:],W) - Y[i] )
    return cost


for iter in range(0, iterations):
    print "\n\nIteration:", iter
    print "\nValue of W before updation is", W
    dJdW = grad(X, Y, W)
    tempW = W - stepsize * (dJdW)

    #print "\nValue of tempW is", tempW
    W = tempW
    print "\nValue of W after updation is", W
    print "the cost value is", cost()






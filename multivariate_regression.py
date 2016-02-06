import numpy as np

### Initalize variables
stepsize = 0.001
iterations = 10
#will be using W0 as b
W = np.ones(6)
W1 = np.empty_like(W)
W1[0] = 132.83
W1[1] = -0.0013
W1[2] = -0.4221
W1[3] = -35.6741
W1[4] = 0.0999
W1[5] = -147.2917
#W1 = [132.83, -0.0013, -0.4221, -35.6741, 0.0999, -147.2917]
print W1.shape
print W1
W = W1


### Read data set and structure it
Data = np.loadtxt("airfoil_self_noise.dat")
columns = Data.shape[1]
Y = Data[:,columns - 1 ]
Features = Data[:,0:(columns - 1)]

print Features.shape
print Y.shape
print W
print W.shape
print Y[0]
print Y[1502]
print Features[0,4]
print Features[1502,0]
print len(Y)
print Features[3,:]

ones = np.ones(Data.shape[0])
X = np.column_stack((ones,Features))
print X.shape
print X
print np.dot(X[3,:],W)


#gradient calculation
def grad(X, Y, W):
    dJdW = np.empty_like(W)
    print "\nValue of dJdW is", dJdW
    for j in range(len(W)):
        sum = 0
        for i in range(len(Y)):
            sum += (Y[i] - (np.dot(X[i,:],W))) * X[i,j]

        print sum
        dJdW[j] = sum/len(Y)
    print "\nValue of dJdW is", dJdW
    return dJdW

#grad descent implementation

for iter in range(1, iterations):
    print "\n\nIteration:", iter
    print "\nValue of W before updation is", W
    dJdW = grad(X, Y, W)
    tempW = W - stepsize * (dJdW)
    print "\nValue of tempW is", tempW
    W = tempW
    print "\nValue of W after updation is", W







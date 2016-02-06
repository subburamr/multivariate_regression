import numpy as np

### Initalize variable
stepsize = 0.0001
W = np.random.randn(5) * 0.01
b = np.random.randn(1) * 0.01
iterations = 10

### Read data set and structure it
Data = np.loadtxt("airfoil_self_noise.dat")

#print data
#x = np.hsplit(data, )
columns = Data.shape[1]

Y = Data[:,columns-1 ]
#print y
X = Data[:,0:(columns - 1)]
#print x

print X.shape
print Y.shape
print W
print W.shape

print b
print b.shape

#grad calculation
def grad(X, Y, W, b):
    return np.empty_like(W), np.empty_like(b)

#grad descent implementation

for iter in range(1, iterations):
    print "\n\nIteration:", iter
    print "\nValue of W before updation is", W
    print "\nValue of b before updation is", b
    dJdW, dJdb = grad(X, Y, W, b)
    tempb = b + stepsize * (-dJdb)
    tempW = W + stepsize * (-dJdW)
    b = tempb
    W = tempW
    print "\nValue of W after updation is", W
    print "\nValue of b after updation is", b






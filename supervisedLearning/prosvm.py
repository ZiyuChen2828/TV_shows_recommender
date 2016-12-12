
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

import numpy as np
from sklearn import datasets
from sklearn.metrics import *
from numpy import loadtxt, ones, zeros, where
import copy

### import some data to play with
##iris = datasets.load_iris()
##X = iris.data[:, :2]  # we only take the first two features
##Y = iris.target
filename = 'data/user_data.dat'
data = loadtxt(filename, delimiter=',')
X = data[:,:] 
filename = 'data/digitsY.dat'
##data = loadtxt(filename, delimiter=',')
newx = []
y = []

#XX = copy.deepcopy(X[4])
#print(XX.shape, XX[31])
for i in range(0, X.shape[0]):
    for j in range(0, X.shape[1]):
        if X[i][j] == 1:                ##to be changed
            XX = copy.deepcopy(X[i])
            XX[j] = 0
            newx.append(XX)
            y.append(j)

X = np.array(newx)
y = np.array(y)
print(X)
print(y)
n,d = X.shape
nTrain = int(n*0.8)


# split the data
Xtrain = X[:nTrain,:]
ytrain = y[:nTrain]
Xtest = X[nTrain:,:]
ytest = y[nTrain:]

print "Training the SVMs..."

C = 1.0  # value of C for the SVMs

# create an instance of SVM with the custom kernel and train it
##myModel = svm.SVC(C = C, kernel=myGaussianKernel)
##myModel.fit(X, Y)

# create an instance of SVM with build in RBF kernel and train it
#equivalentGamma = 1.0 / (2 * _gaussSigma ** 2)
model = svm.SVC(C = C, kernel='rbf', gamma=0.5)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
print ""
print "Testing the SVMs..."

h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


# get predictions for both my model and true model
##myPredictions = myModel.predict(np.c_[xx.ravel(), yy.ravel()])
##myPredictions = myPredictions.reshape(xx.shape)

accuracy = accuracy_score(ytest, ypred)
precision = precision_score(ytest, ypred)
recall = recall_score(ytest, ypred)

print "Neural Network Accuracy = "+str(accuracy)
print "Neural Network Precision = "+str(precision)
print "Neural Network Recall = "+str(recall)



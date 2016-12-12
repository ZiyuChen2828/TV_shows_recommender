
import numpy as np
from sklearn import datasets
from sklearn.metrics import *
from numpy import loadtxt, ones, zeros, where
import copy

from newnn1 import NeuralNet

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
y = np.array([y]).T
print(X)
print(y)
            
n,d = X.shape



nTrain = int(n*0.8)


# split the data
Xtrain = X[:nTrain,:]
ytrain = y[:nTrain]
Xtest = X[nTrain:,:]
ytest = y[nTrain:]

# train the decision tree
model = NeuralNet(layers=np.array([25]),epsilon=0.12, learningRate = 2, numEpochs=100)
model.ff(X,y)

model.fit(Xtrain,ytrain)


# output predictions on the remaining data
ypred = model.predict(Xtest, ytest)


# compute the training accuracy of the model
accuracy = accuracy_score(ytest, ypred)
precision = precision_score(ytest, ypred)
recall = recall_score(ytest, ypred)

print "Neural Network Accuracy = "+str(accuracy)
print "Neural Network Precision = "+str(precision)
print "Neural Network Recall = "+str(recall)

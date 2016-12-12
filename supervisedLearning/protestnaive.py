
print(__doc__)

import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import datasets
from sklearn.metrics import *
from numpy import loadtxt, ones, zeros, where
import copy

from pronaiveBayes import NaiveBayes

# load the data set
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
#y = np.array([y]).T

#print(X)
#print(y)
n,d = X.shape
nTrain = 0.5*n  #training on 50% of the data

# shuffle the data
##idx = np.arange(n)
##np.random.seed(13)
##np.random.shuffle(idx)
##X = X[idx]
##y = y[idx]

nTrain = int(n*0.8)


# split the data
Xtrain = X[:nTrain,:]
ytrain = y[:nTrain]
Xtest = X[nTrain:,:]
ytest = y[nTrain:]

# train the decision tree
##modelDT = DecisionTreeClassifier()
##modelDT.fit(Xtrain,ytrain)

# train the naive Bayes
modelNB = NaiveBayes(useLaplaceSmoothing=True)
modelNB.ff(X,y)
modelNB.fit(Xtrain,ytrain)

# output predictions on the remaining data
##ypred_DT = modelDT.predict(X,y)
ypred = modelNB.predict(Xtest, ytest)



# compute the training accuracy of the model
#accuracyDT = accuracy_score(ytest, ypred_DT)
#accuracyNB = accuracy_score(ytest, ypred_NB)

#print "Decision Tree Accuracy = "+str(accuracyDT)
#print "Naive Bayes Accuracy = "+str(accuracyNB)

accuracy = accuracy_score(ytest, ypred)
precision = precision_score(ytest, ypred)
recall = recall_score(ytest, ypred)

print "Naive Bayes Accuracy = "+str(accuracy)
print "Naive Bayes Precision = "+str(precision)
print "Naive Bayes Recall = "+str(recall)

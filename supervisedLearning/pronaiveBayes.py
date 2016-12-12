
import numpy as np

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing
      
    def ff(self, X, y):

        n,d = X.shape
        sety = set(y)
        self.sety = sety
        count = 347
        k = np.zeros(count)

        for i in range(0, n):
            k[y[i]] += 1  #number of instance y = yj

        pro = k/np.sum(k)
        self.pro = pro
        
    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,d = X.shape
##        sety = set(y)
##        self.sety = sety
        #m = {}
##        count = 0
##        for i in range(0, n):
##            if y[i] not in m:
##                count += 1
##                #m[i] = y[i]  #count type K
        #count = len(sety)
##        count = 347
##        k = np.zeros(count)
##
##        for i in range(0, n):
##            k[y[i]] += 1  #number of instance y = yj
##
##        pro = k/np.sum(k)
##        self.pro = pro
        count = 347
        pro = self.pro
        

        b = np.zeros((count, d))
        a = np.zeros(count)
        theta = np.zeros((count, d)) 

        # calculate theta
        for i in range(0, n):
            for j in range(0, d):
                    b[y[i]][j] += X[i][j]
     
        a = np.sum(b, axis=1)

        for i in range(0, count):
            for j in range(0, d):
                theta[i][j] = (b[i][j] + 1)/(a[i] + d)

        self.theta = theta
        
    def predict(self, X, z):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''

        n,d = X.shape
        y = np.zeros(n)
        for i in range(0, n):
            bes = np.log(self.pro[z[i]]) + np.dot(X[i], np.log(self.theta[z[i]]))
            c = 0
            for j in self.sety:
                s = np.log(self.pro[j]) + np.dot(X[i], np.log(self.theta[j]))
                if s > bes:
                    c += 1
            if c <= 1:
                y[i] = z[i]
            else:
                y[i] = -1
        return y

    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        n,d = X.shape
        y = []
        count = len(self.sety)
        a = np.zeros((n, count))
        prob = np.zeros((n, count))
        for i in range(0, n):
            bes = -99999999999
            c = 0
            for j in self.sety:
                s = np.log(self.pro[j]) + np.dot(X[i], np.log(self.theta[j]))
                a[i][j] = s
                if s > bes:
                    bes = s
                    c = j
            a[i] -= bes
            a[i] = np.exp(a[i])
            sum = 0
            for j in self.sety:
                sum += a[i][j]
            prob[i] = a[i]/sum

        return prob
        
        
        
class OnlineNaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing
        self.theta1 = {}
        self.theta2 = {}
        self.theta = {}
        self.pro1 = {}
        self.pro2 = 0
        self.pro = {}
        self.sety = set()

        
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,d = X.shape

        pro1 = self.pro1
        for i in range(0, n):
            if y[i] not in pro1:
                pro1[y[i]] = 1
            else:
                pro1[y[i]] += 1

        self.pro2 += n

        for i in pro1:
            self.pro[i] = pro1[i]/float(self.pro2)

##        pro = self.pro
##        for i in pro1:
##            pro[i] = pro1[i] / float(self.pro2)
##        self.pro = pro

        self.pro1 = pro1

        sety = self.sety
        for i in set(y):
            sety.add(i)

        theta1 = self.theta1
        theta2 = self.theta2
        theta = self.theta

        for i in range(0, n):
            if y[i] not in theta1:
                theta1[y[i]] = np.ones(d)
                for j in range(0, d):
                    theta1[y[i]][j] += X[i][j]
            else:
                for j in range(0, d):
                    theta1[y[i]][j] += X[i][j]           

            for j in range(0, d):
                if y[i] not in theta2:
                    theta2[y[i]] = d + X[i][j]
                else:
                    theta2[y[i]] += X[i][j]

        for i in sety:
            theta[i] = theta1[i]/theta2[i]

        self.theta1 = theta1
        self.theta2 = theta2
        self.theta = theta
        


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n,d = X.shape
        y = np.zeros(n)
        for i in range(0, n):
            bes = -99999999999
            c = 0
            for j in self.sety:
                s = np.log(self.pro[j]) + np.dot(X[i], np.log(self.theta[j]))
                if s > bes:
                    bes = s
                    c = j
            y[i] = c
        return y
    
    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        n,d = X.shape
        y = []
        count = len(self.sety)
        a = np.zeros((n, count))
        prob = np.zeros((n, count))
        for i in range(0, n):
            bes = -99999999999
            c = 0
            for j in self.sety:
                s = np.log(self.pro[j]) + np.dot(X[i], np.log(self.theta[j]))
                a[i][j] = s
                if s > bes:
                    bes = s
                    c = j
            a[i] -= bes
            a[i] = np.exp(a[i])
            sum = 0
            for j in self.sety:
                sum += a[i][j]
            prob[i] = a[i]/sum

        return prob
        



        

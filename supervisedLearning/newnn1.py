
import numpy as np
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm 


class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate = 2, numEpochs=100):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
      

    def ff(self, X, y):

        n,d = X.shape
        count = 0
        m = []

        for i in range(0, n):
            if y[i] not in m:
                m.append(y[i])
                count += 1
##        sety = set(y)
##        count = len(sety)
        self.count = count
        self.m = m


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        
        def forwardpropagation(thetalist, alist, X, newlayers): #handle one instance
            #return an array of (1*d)
            n = X.shape[0]
            X = X.T
            alist.append(np.concatenate((np.ones((1)), X), axis = 0))
            for i in range(0, len(newlayers) - 1):
                X = np.concatenate((np.ones((1)), X), axis = 0)
##                theta1 = np.reshape(theta[0:newlayers[i+1] * (newlayers[i] + 1)],
##                        (newlayers[i+1], newlayers[i] + 1))
##                thetalist.append(theta1)
##                theta = np.delete(theta, np.s_[0:newlayers[i+1] * (newlayers[i] + 1)])
                
                z = np.dot(thetalist[i], X)
                a = 1/(np.exp(-z) + 1)
                if i != len(newlayers) - 2:
                    alist.append(np.concatenate((np.ones((1)), a), axis = 0))
                else:
                    alist.append(a)  #an array of (d*1)
                #alist.append(np.concatenate((np.ones((1)), a), axis = 0))
                X = a
            return X.T


        def cost(thetalist, lam, newy, hx):
            n,d = newy.shape
            sum = 0
            for i in range(0, n):
                for k in range(0, d):
                    if newy[i][k] == 1:
                        sum += np.log(hx[i][k])
                    else:
                        sum += np.log(1 - hx[i][k])
            sumtheta = 0
            for i in range(0, len(thetalist)):
                n,d = thetalist[i].shape
                for j in range(0, n):
                    for h in range(0, d):
                        sumtheta += thetalist[i][j][h] * thetalist[i][j][h]
            cost = (float(sumtheta*lam/2.0) - float(sum))/n
            return cost
    
        def gradient(thetalist, alist, newlayers, newy, l):  #l=0,1,2,...
            deltalist = [] #an array of (d*1)
            #print(alist)
            deltalist.append(alist[-1] - newy.T)
            for i in range(0, len(newlayers) - 2):
                deltalist.append(np.dot(thetalist[-i-1].T,
                                deltalist[i]) * alist[-i-2] * (1-alist[-i-2]))
            temp = []
            #print(thetalist[-i-1].T.shape)
            #print(deltalist[i].shape)
            for i in range(0, len(deltalist)):
                temp.append(deltalist[-i-1])  #reverse deltalist
            deltalist = temp
            #print(deltalist[1].T.shape)
            #print(alist[1].shape)
            #print(np.dot(np.array([deltalist[0]]).T, np.array([alist[0]])).shape)
            return np.dot(np.array([deltalist[l]]).T, np.array([alist[l]]))
        n,d = X.shape
        lam = 0.000001
        count = self.count
        m = self.m
        newy = np.zeros((n,count))
        for i in range(0, n):
            for j in range(0, count):
                if y[i] == m[j]:
                    newy[i][j] = 1
                else:
                    newy[i][j] = 0    #create newy
        
        b = np.append(d, self.layers)
        newlayers = np.append(b, count)
        self.newlayers = newlayers   #initialize newlayers
        L = len(newlayers)
##        theta = np.zeros((1, newlayers[1] * (newlayers[0] + 1)))
        

        thetalist = []
        for i in range(0, L-1):    #initialize thetalist
            thetalist.append(np.zeros((newlayers[i+1], newlayers[i]+1)))
        for i in range(0, L-1):
            nn,dd = thetalist[i].shape
            for j in range(0, nn):
                for h in range(0, dd):
                    thetalist[i][j][h] = np.random.uniform(-self.epsilon, self.epsilon)
                    
##        for i in range(1, L-1):
##            b = np.zeros((1, newlayers[i+1] * (newlayers[i] + 1)))
##            theta = np.append(theta, b)          
##        for i in range(0, len(theta)):
##            theta[i] = np.random.uniform(-self.epsilon, self.epsilon) #initialize

        for i in range(0, self.numEpochs):
            Dlist = []
            for j in range(0, L-1):
                Dlist.append(np.zeros((newlayers[j+1], newlayers[j]+1)))
            for k in range(0, n):
                alist = [] 
                forwardpropagation(thetalist, alist, X[k,:], newlayers)
                for l in range(0, L-1):
                    pp, qq = Dlist[l].shape
                    aa = gradient(thetalist, alist, newlayers, newy[k,:], l)
                    bb = aa[1:]
                    if l!= L-2:
                        Dlist[l] += bb
                    else:
                        Dlist[l] += aa
##                    for p in range(0, pp):
##                        for q in range(0, qq):
##                            if l!= L-2:
##                                Dlist[l][p][q] += aa[p+1][q]
##                            else:
##                                Dlist[l][p][q] += aa[p][q]
                    #Dlist[l] += gradient(thetalist, alist, newlayers, newy[k,:], l)

            
            for h in range(0, L-1):
                Dlist[h] = Dlist[h]/n
                nn, dd = Dlist[h].shape
                for z in range(0, nn):
                    for c in range(0, dd):
                        if c != 0:
                            Dlist[h][z][c] += lam * thetalist[h][z][c]

##            for h in range(0, L-1):
##                nn, dd = Dlist[h].shape
##                for z in range(0, nn):
##                    for c in range(0, dd):
##                        if c != 0:
##                            Dlist[h][z][c] = Dlist[h][z][c]/n
##                        else:
##                            Dlist[h][z][c] = Dlist[h][z][c]/n + lam * thetalist[h][z][c]
            
            for h in range(0, L-1):
                nn, dd = thetalist[h].shape
                for z in range(0, nn):
                    for c in range(0, dd):
                        thetalist[h][z][c] = thetalist[h][z][c] - self.learningRate * Dlist[h][z][c]
            #print(i)
            # not wrote converge
        self.thetalist = thetalist
    def predict(self, X, z):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        def forwardpropagation(thetalist, alist, X, newlayers): #handle one instance
            #return an array of (1*d)
            
            n = X.shape[0]
            XX = X.T
            alist.append(np.concatenate((np.ones((1)), XX), axis = 0))
            for i in range(0, len(newlayers) - 1):
                XX = np.concatenate((np.ones((1)), XX), axis = 0)
##                theta1 = np.reshape(theta[0:newlayers[i+1] * (newlayers[i] + 1)],
##                        (newlayers[i+1], newlayers[i] + 1))
##                thetalist.append(theta1)
##                theta = np.delete(theta, np.s_[0:newlayers[i+1] * (newlayers[i] + 1)])
                z = np.dot(thetalist[i], XX)
                a = 1/(np.exp(-z) + 1)
                if i != len(newlayers) - 2:
                    alist.append(np.concatenate((np.ones((1)), a), axis = 0))
                else:
                    alist.append(a)  #an array of (d*1)
                XX = a
            return XX.T
        n,d = X.shape
        y = np.zeros(n)   
        for i in range(0, n):
            c = 0
            alist=[]
            yy = forwardpropagation(self.thetalist, alist, X[i,:], self.newlayers)
            a = self.m.index(z[i])
            k = 0
            for j in range(0, self.count): 
                if yy[a] <= yy[j]:
                    c += 1
            if c <= 1:
                y[i] = z[i]
            else:
                y[i] = -1
        return y

    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        for p in range(0, 25):
            a = np.reshape(self.thetalist[0][p][1:], (20,20))
            m = []
            for i in range(0, 20):
                for j in range(0, 20):
                    a[i][j] = (a[i][j] + 1) * 255//2
                    m.append(a[i][j])
            #print(a)
            img = Image.new( 'L', (20,20)) # create a new black image
            plt.subplot(5,5,p+1)
            #plt.imshow(a, cmap="Greys_r")
            print(m)
            img.putdata(m)
            #img.show()
            plt.imshow(img, cmap=cm.gray)
            plt.axis("off")
    ##        pixels = img.load() # create the pixel map
    ##
    ##        for i in range(img.size[0]):    # for every pixel:
    ##            for j in range(img.size[1]):
    ##                pixels[i][j] = a[i][j] # set the colour accordingly
    ##        print(1)
        plt.show()
        #img.save('d:/dog.jpg')

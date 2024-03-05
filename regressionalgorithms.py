import numpy as np
import math
import matplotlib.pyplot as plt
import MLCourse.utilities as utils
import script_regression as sr
import MLCourse.plotfcns as plotfcns
import time

# -------------
# - Baselines -
# -------------

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.weights = None

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        # Most regressors return a dot product for the prediction
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.min = 0
        self.max = 1

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__(self, parameters = {}):
        self.params = parameters
        self.mean = None

    def learn(self, Xtrain, ytrain):
        # Learns using the traindata
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'regwgt': 0.0,
            'features': [1,2,3,4,5],
        }, parameters)

    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        numfeatures = Xless.shape[1]

        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        #ordinary inverse function
        #self.weights = np.linalg.inv(inner).dot(Xless.T).dot(ytrain) / numsamples
        # generalized pseudo inverse function (Ans to the Q. 2(a))
        self.weights = np.linalg.pinv(inner).dot(Xless.T).dot(ytrain) / numsamples
        
    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

# ---------
# - TODO: -
# ---------
#(Ans to the Q. 2(c))
class RidgeLinearRegression(Regressor): 
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items({'regwgt': 0.5}, parameters)
    
    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain
        numfeatures = Xless.shape[1]

        inner = (Xless.T.dot(Xless) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(Xless.T).dot(ytrain) / numsamples
        #self.weights = np.linalg.pinv(inner).dot(Xless.T).dot(ytrain) / numsamples

    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest
    
# Ans. to the Q. no - 2(d)
class LassoRegression(Regressor): 
    """
    Linear Regression with Lasso regularization (l1 regularization) and 
    Implemented with soft thresholding operator(proximal method)
    """
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = utils.update_dictionary_items({'regwgt': 0.5}, parameters)
    
    def proximal_operator(self, prox_parameter, stepsize, regwgt):
        prox_parameter_dim=prox_parameter.shape[0]
        for i in range(prox_parameter_dim):
            if prox_parameter[i] > stepsize*regwgt:
                self.weights[i]= prox_parameter[i]-stepsize*regwgt
            elif np.absolute(prox_parameter[i]) <= stepsize*regwgt:
                self.weights[i]=0
            elif prox_parameter[i] < -(stepsize*regwgt):
                self.weights[i]=prox_parameter[i]+stepsize*regwgt
    
    
    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain
        numfeatures = Xless.shape[1]
        #set the weight vector with zeros
        self.weights= np.zeros([numfeatures,])
        #set error to be infinity and tolerance to be 10e-4 initially
        err = float('inf')
        tolerance= 10e-4
        
        #Precomputing this matrices to avoid recomputing later
        XX = np.dot(Xtrain.T,Xtrain)/numsamples
        Xy = np.dot(Xtrain.T,ytrain)/numsamples
        
        #stepsize is calculated by deriving the frobenius norm of the matrix XX
        stepsize= 1/(2*(np.linalg.norm(XX)))
        #c_w calculation
        c_w= sr.geterror(np.dot(Xtrain,self.weights),ytrain)
        while np.absolute(c_w-err)> tolerance:
            err= c_w
            #proximal operator projexts back into the space of sparse solutions given by l1 regularizer
            prox_parameter= np.add(np.subtract(self.weights,stepsize*np.dot(XX,self.weights)), stepsize*Xy)
            self.proximal_operator(prox_parameter,stepsize,self.params['regwgt'])
            #update c_w due to update in weights value from the proximal_operator function
            l1norm_weight=np.linalg.norm(self.weights,ord=1)
            c_w=sr.geterror(np.dot(Xtrain,self.weights),ytrain) + self.params['regwgt']*l1norm_weight

    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest
    
# Ans. to the Q. no - 2(e) 1st part
class StochasticGradientDescent(Regressor): 
    """
    Linear Regression with Lasso regularization (l1 regularization) and 
    Implemented with soft thresholding operator(proximal method)
    """
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        #self.params = utils.update_dictionary_items({'regwgt': 0.5}, parameters)
        self.params={}
        self.numruns=5
        self.yaxis= np.zeros(1000)  #to be used for storing value in order to plot graph
        self.yaxis1=np.zeros(1000)
        
        
    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        #Xless = Xtrain[:, self.params['features']]
        Xless = Xtrain
        numfeatures = Xless.shape[1]
        #set the weight vector with zeros
        #self.weights= np.zeros([numfeatures,])
        self.weights=np.random.random(numfeatures)
        #setting epochs and stepsize with it's initial value as of question 2(e)
        self.times=[]
        epochs=1000
        initial_stepsize=0.01
        start=time.time()
        for i in range(epochs):
            
            #shuffling data points from 1 to the number of samples n
            datapoints_array=np.arange(numsamples)
            np.random.shuffle(datapoints_array)
            for j in range(numsamples):
                gradient=np.dot(np.subtract(np.dot(Xtrain[datapoints_array[j]].T,self.weights), ytrain[datapoints_array[j]]), Xtrain[datapoints_array[j]])
                #For convergence, the stepsize needs to decrease with time
                stepsize= initial_stepsize/(1+i)
                self.weights=np.subtract(self.weights, stepsize*gradient)
             
            #storing error to plot the graph for a step size=0.01 and epochs=1000
            #Ans to the Q.no 2(e) 2nd part
            self.yaxis[i]= self.yaxis[i]+sr.geterror(np.dot(Xtrain,self.weights),ytrain)
            self.times.append(time.time() - start)
            k = int(self.times[i])
            self.yaxis1[i]=self.yaxis1[i]+k
        x = np.arange(1000)
        plt.plot(x, self.yaxis)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        #Ans to the Q. no - 2(f) 2nd part
        #Reporting the error vs epoch for a stochastic gradient descent with stepsize-0.01 & epochs=1000
        plt.show()
        
        #print("\n")
        #print(self.times)
        #print("\n")
        plt.plot(self.yaxis1, self.yaxis)
        plt.xlabel('Runtime')
        plt.ylabel('Error')
        #Reporting the error vs runtime for a stochastic gradient descent with stepsize-0.01 & epochs=1000
        plt.show()
        
    
    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest
    
# Ans. to the Q. no - 2(f) 1st part
#Batch gradient descent with line search
class BatchGradientDescent(Regressor): 
    """
    Linear Regression with Lasso regularization (l1 regularization) and 
    Implemented with soft thresholding operator(proximal method)
    """
    def __init__(self, parameters = {}):
        # Default parameters, any of which can be overwritten by values passed to params
        #self.params = utils.update_dictionary_items({'regwgt': 0.5}, parameters)
        self.params={}
        self.numruns=5
        self.yaxis= np.zeros(5000)  #to be used for storing value in order to plot graph
        self.xaxis= np.arange(5000)
    
    def lineSearch(self,Xtrain,ytrain,weight_t,gradient,cost):
        numsamples = Xtrain.shape[0]
        #optimization parameters:
        maxi_stepsize=1.0
        #for t=0.5, the stepsize reduces more quickly according to the notes
        t=0.5
        tolerance= 10e-4
        stepsize=maxi_stepsize
        weight= weight_t
        objective=cost
        max_iteration=100
        i=0
        
        while i<max_iteration:
            weight=weight_t-stepsize*gradient
            i=i+1
            #Ensure improvement is atleast as much as tolerance
            if cost< objective-tolerance:
                break
            else:
                #the objective is worse and so we decrease stepsize
                stepsize=t*stepsize
            cost=sr.geterror(np.dot(Xtrain,weight),ytrain)
        if i==max_iteration:
            #could not improve solution
            stepsize=0
            #weight_t=0
            return stepsize
        return stepsize
    def learn(self, Xtrain, ytrain):
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        #Xless = Xtrain[:, self.params['features']]
        Xless = Xtrain
        numfeatures = Xless.shape[1]
        stepsize=0.01
        #set the weight vector with zeros
        #self.weights= np.zeros([numfeatures,])
        self.weights=np.random.random(numfeatures)
        
        err = float('inf')
        tolerance= 10e-4
        max_iteration=10e5
        i=0
        c_w=sr.geterror(np.dot(Xtrain,self.weights),ytrain)
        while (np.absolute(c_w-err) > tolerance) and (i<max_iteration):
            err=c_w
            g=np.dot(Xtrain.T,np.subtract(np.dot(Xtrain,self.weights),ytrain))/numsamples
            #stepsize is chosen by line search
            stepsize=self.lineSearch(Xtrain,ytrain,self.weights,g,c_w)
            self.weights=self.weights-stepsize*g
            c_w=sr.geterror(np.dot(Xtrain,self.weights),ytrain)
            
            if(i<5000):
                self.yaxis[i]=self.yaxis[i]+err
            i=i+1
        #print("\n")
        #print(i)
        #print("\n")
        
        #plt.plot(self.xaxis, self.yaxis/self.numruns)
        #plt.show()
        #self.data1(self.xaxis, self.yaxis/self.numruns)
        x = np.arange(5000)
        plt.plot(x, self.yaxis)
        plt.xlabel('Convergence')
        plt.ylabel('Error')
        plt.show()
        
    def predict(self, Xtest):
        Xless = Xtest
        ytest = np.dot(Xless, self.weights)
        return ytest

#Ans to the Bonus Q. (a)
class Momentum(Regressor):
    def __init__(self, parameters={}):
        self.params = {}

        """initialize moving average as vector of zeros"""
        self.m_t  = np.zeros([385,])
        self.beta = 0.9
        self.alpha = 0.01


    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros([385, ])
        t=0

        while (t<1000):					#till 1000 iterations
            t+=1
            for j in range(numsamples):
                g_t = np.dot(np.subtract(np.dot(Xtrain[j].T, self.weights), ytrain[j]), Xtrain[j])
                m_t = self.beta* self.m_t + (1-self.beta)*g_t	#updates the moving averages of the gradient
                m_cap = m_t/(1-(self.beta**t))		#calculates the bias-corrected estimates
                weight_prev = self.weights
                self.weights = weight_prev - (self.alpha*m_cap)	 #updates the parameters
            
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest

#Ans to the Bonus Q. (b) 
class ADAM(Regressor):

    def __init__(self, parameters={}):
        self.params = {}

        """initialize moving average as vector of zeros"""
        self.m_t  = np.zeros([385,])
        self.v_t = np.zeros([385,])

        self.alpha = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.999  # initialize the values of the parameters
        self.epsilon = 1e-8


    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros([385, ])
        t=0
        conv = 0
        while (t<1000):					#till 1000 iterations
            t+=1
            for j in range(numsamples):
                g_t = np.dot(np.subtract(np.dot(Xtrain[j].T, self.weights), ytrain[j]), Xtrain[j])
                m_t = self.beta_1* self.m_t + (1-self.beta_1)*g_t
                v_t = self.beta_2* self.v_t + (1-self.beta_2)*np.dot(g_t.T,g_t)
                m_cap = m_t/(1-(self.beta_1**t))		
                v_cap = v_t/(1-(self.beta_2**t))
                weight_prev = self.weights
                self.weights = weight_prev - (self.alpha*m_cap)/(np.sqrt(v_cap)+self.epsilon)	#updates the parameters

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest
    
#Ans to the Bonus Q. (c)
class MomentumWithoutInitializationBias(Regressor):
    def __init__(self, parameters={}):
        self.params = {}

        """initialize moving average as vector of zeros"""
        self.m_t  = np.zeros([385,])
        self.v_t = np.zeros([385,])

        self.alpha = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.999  # initialize the values of the parameters
        self.epsilon = 1e-8


    def learn(self, Xtrain, ytrain):
        numsamples = Xtrain.shape[0]
        self.weights = np.zeros([385, ])
        t=0
        conv = 0
        while (t<1000):					#till 1000 iterations
            t+=1
            for j in range(numsamples):
                g_t = np.dot(np.subtract(np.dot(Xtrain[j].T, self.weights), ytrain[j]), Xtrain[j])
                m_t = self.beta_1* self.m_t + (1-self.beta_1)*g_t
                v_t = self.beta_2* self.v_t + (1-self.beta_2)*np.dot(g_t.T,g_t)
                weight_prev = self.weights
                self.weights = weight_prev - (self.alpha*m_t)/(np.sqrt(v_t)+self.epsilon)	#updates the parameters

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        return ytest
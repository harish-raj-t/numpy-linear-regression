#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class LinearRegression:
    def __init__(self,learning_rate=0.1, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        #X dimensions are m*n 
        #m -> number of instances, n-> features
        n_instances, n_features = X.shape
        self.weights =  np.zeros(n_features)
        self.bias = 0
        

        for _ in range(self.epochs):
            output = np.dot(X,self.weights) + self.bias
            error = output-y
            dw = np.dot(X.T,error)/ n_instances
            db = np.sum(error) / n_instances 
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            output = X @ self.weights + self.bias
            error = output-y
            
        
    def predict(self,X):
        return np.dot(X,self.weights) + self.bias


# In[12]:


class LinearRegressionMiniSGD:
    def __init__(self,learning_rate=0.1, epochs = 1000, batches = 64):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batches = batches
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        #X dimensions are m*n 
        #m -> number of instances, n-> features
        n_instances, n_features = X.shape
        self.weights =  np.zeros(n_features)
        self.bias = 0
        

        for _ in range(self.epochs):
            indices = np.random.permutation(n_instances)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0,n_instances,self.batches):
                X_batch = X_shuffled[i:i+self.batches]
                y_batch = y_shuffled[i:i+self.batches]
                output = np.dot(X_batch,self.weights) + self.bias
                error = output-y_batch
                dw = np.dot(X_batch.T,error)/ n_instances
                db = np.sum(error) / n_instances 
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
        
    def predict(self,X):
        return np.dot(X,self.weights) + self.bias


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n_samples = 3        # Number of data points
n_features = 2         # Two features
true_coefficients = np.array([2.5, -1.7])  # Coefficients for each feature
intercept = 4.0        # Intercept
noise_level = 1.0      # Standard deviation of the noise

X = np.random.rand(n_samples, n_features) * 10

y = X @ true_coefficients + intercept + np.random.normal(0, noise_level, size=n_samples)


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[16]:


lr = LinearRegressionMiniSGD(learning_rate = 0.1)
lr.fit(X,y)


# In[17]:


pred = lr.predict(X)


# In[18]:


loss = np.sqrt(np.mean(np.square(pred-y)))
loss


# In[20]:


x_test = scaler.transform([[8.84975613, 2.81881006]])
lr.predict(x_test)


# In[ ]:





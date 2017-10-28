#import statsmodels.api as sm
import pandas as pd #not of your use
import numpy as np
import logging
import json

logging.basicConfig(filename='output.log',level=logging.DEBUG)


#utility functions
def loadData(file_name):
    df = pd.read_csv(file_name)
    logging.info("Number of data points in the data set "+str(len(df)))
    y_df = df['output']
    keys = ['company_rating','model_rating','bought_at','months_used','issues_rating','resale_value']
    X_df = df.get(keys)
    
    return X_df, y_df

def accuracy(X, y, model):
    
    thetas = np.array(model['theta'],dtype=np.float128)
    print thetas
    y_predicted = predict(X,thetas)
    y_predicted = np.floor(0.3 + y_predicted)
    
    numerator = np.sum(np.logical_not(np.logical_xor(y.astype(bool),y_predicted.astype(bool))))
    relative_error = numerator/(1.0*len(X))
    acc = relative_error * 100.0
    
    print "Accuracy associated with this model is "+str(acc)

def predict(X,theta):
	z =  np.dot(X,theta)
	return sigmoid(z)

def sigmoid(x):
	return 1/(1 + np.exp(-x))

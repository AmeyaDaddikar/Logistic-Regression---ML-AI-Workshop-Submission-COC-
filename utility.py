import numpy as np
import pandas as pd #not of your use
import logging
import json


logging.basicConfig(filename='log/output.log',level=logging.DEBUG)


#utility functions
def loadData(file_name):
    df = pd.read_csv(file_name)
    logging.info("Number of data points in the data set "+str(len(df)))
    y_df = df['output']
    keys = ['company_rating','model_rating','bought_at','months_used','issues_rating','resale_value']
    X_df = df.get(keys)
    
    return X_df, y_df


def normalizeData(X_df, y_df, model):
    #save the scaling factors so that after prediction the value can be again rescaled
    model['input_scaling_factors'] = [list(X_df.mean()),list(X_df.std())]
    X = np.array((X_df-X_df.mean())/X_df.std())
    
    return X, y_df, model

def normalizeTestData(X_df, y_df, model):
    meanX = model['input_scaling_factors'][0]
    stdX = model['input_scaling_factors'][1]

    X = 1.0*(X_df - meanX)/stdX

    return X, y_df


def accuracy(X, y, model):

    y_predicted = predict(X,np.array(model['theta']))
    y_predicted = np.floor(0.5 + y_predicted)
    
    numerator = np.sum(np.logical_not(np.logical_xor(y.astype(bool),y_predicted.astype(bool))))
    relative_error = numerator/(1.0*len(X))
    acc = relative_error * 100.0
	
    print "Accuracy associated with this model is "+str(acc)

def predict(X,theta):
	z =  np.dot(X,theta)
	return sigmoid(z)

def sigmoid(x):
	return 1/(1 + np.exp(-x))

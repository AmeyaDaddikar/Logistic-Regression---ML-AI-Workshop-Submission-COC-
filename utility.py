

import numpy as np
import pandas as pd #not of your use
import logging
import json

FILE_NAME_TRAIN = 'Resources/train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'Resources/test.csv' #replace
ALPHA = 1e-3
EPOCHS = 100000
MODEL_FILE = 'models/model2'
train_flag = True

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

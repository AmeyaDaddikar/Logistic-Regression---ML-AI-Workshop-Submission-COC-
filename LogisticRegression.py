import numpy as np
import logging
import json
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'data/combined_test.csv'
FILE_NAME_TEST = 'data/test.csv'
MODEL_FILE = 'models/final_model'

logging.basicConfig(filename='log/output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)

###########FUNCTIONS################################################################

#this function appends 1 to the start of the input X and returns the new array
def appendIntercept(X):
	ones_vec = np.ones((X.shape[0],1))
	return np.hstack((ones_vec,X))
    
def train(X, y, model):
	m = len(y)

	logit_mod = sm.Logit(y, X)
	logit_res = logit_mod.fit(disp=0)
	
	#Saves the thetas in the JSON file
	model['theta'] = list(logit_res.params)
	
	return model
	
########################main function###########################################
def main():

    #TRAINS THE MODEL
    print FILE_NAME_TRAIN
    model = {}
    X,y = loadData(FILE_NAME_TRAIN)
    X = appendIntercept(X)
    model = train(X, y, model)
    with open(MODEL_FILE,'w') as f:
        f.write(json.dumps(model))

    #TESTS THE MODEL
    model = {}
    with open(MODEL_FILE,'r') as f:
        model = json.loads(f.read())
        X,y = loadData(FILE_NAME_TEST)
        X = appendIntercept(X)
        accuracy(X,y,model)

if __name__ == '__main__':
    main()

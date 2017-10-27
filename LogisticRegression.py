
import numpy as np
import logging
import json
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'Resources/train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'Resources/test.csv' #replace
ALPHA = 7.9e-3
EPOCHS = 50000#keep this greater than or equl to 5000 strictly otherwise you will get an error
MODEL_FILE = 'models/model1'
train_flag = False

logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
#################################################################################################
#####################################write the functions here####################################
#################################################################################################
#this function appends 1 to the start of the input X and returns the new array
def appendIntercept(X):
	ones_vec = np.ones((X.shape[0],1))
	return np.hstack((ones_vec,X))
    




 #intitial guess of parameters (intialize all to zero)
 #this func takes the number of parameters that is to be fitted and returns a vector of zeros
def initialGuess(n_thetas):
	return np.zeros(n_thetas)



def train(theta, X, y, model):
	J = [] #this array should contain the cost for every iteration so that you can visualize it later when you plot it vs the ith iteration
     #train for the number of epochs you have defined
	m = len(y)
     #your  gradient descent code goes here
     #steps
     #run you gd loop for EPOCHS that you have defined
        #calculate the predicted y using your current value of theta
        # calculate cost with that current theta using the costFunc function
        #append the above cost in J
        #calculate your gradients values using calcGradients function
        # update the theta using makeGradientUpdate function (don't make a new variable assign it back to theta that you received)
	for i in range (0,EPOCHS):
		#print str(i) + "\b\b\b\b"
		predicted_y = predict(X,theta)
		#error_cost  = costFunc(m,y,predicted_y)
		#J.append(error_cost)
		new_gradients = calcGradients(X,y,predicted_y,m)
		theta = makeGradientUpdate(theta,new_gradients)

	model['J'] = J
	model['theta'] = list(theta)
	return model

	
#this function will calculate the total cost and will return it
def costFunc(m,y,y_predicted):

	log_h = np.log(y_predicted)
	one_minus_log_h = np.log(1-y_predicted)
	cost_arr = np.add(np.multiply(y,log_h),np.multiply(1-y,one_minus_log_h))
	cost = np.sum(cost_arr)
	cost /= (-m)
	return cost	

def calcGradients(X,y,y_predicted,m):
	difference = np.subtract(y_predicted,y)
	difference = difference.values.reshape((X.shape[0],1))
	
	summation = np.multiply(X,difference)
	
	return np.sum(summation,axis=0)/m
    #apply the formula , this function will return cost with respect to the gradients
    # basically an numpy array containing n_params

#this function will update the theta and return it
def makeGradientUpdate(theta, grads):
    return np.subtract(theta,ALPHA*grads)


#this function will take two paramets as the input
def predict(X,theta):
	z =  np.dot(X,theta)
	return sigmoid(z)

def sigmoid(x):
	return 1/(1 + np.exp(-x))

########################main function###########################################
def main():
    if(train_flag):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)
        with open(MODEL_FILE,'w') as f:
            f.write(json.dumps(model))

    else:
        model = {}
        with open(MODEL_FILE,'r') as f:
            model = json.loads(f.read())
            X_df, y_df = loadData(FILE_NAME_TEST)
            X,y = normalizeTestData(X_df, y_df, model)
            X = appendIntercept(X)
            accuracy(X,y,model)

if __name__ == '__main__':
    main()

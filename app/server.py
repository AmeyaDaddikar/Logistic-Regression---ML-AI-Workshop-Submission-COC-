
from flask import Flask, render_template, request
import json
from flask_cors import CORS
import numpy as np
from math import exp

app = Flask(__name__)
CORS(app)

PHONES_INFO = {"Apple":['Iphone 4', 'Iphone 4s', 'Iphone 5', 'Iphone 5s', 'Iphone 6'], "Motorola":['G1', 'G2', 'G3'], "OnePlus":['One','Two', 'X'],
"Samsung":['Galaxy y', 'Galaxy Win', 'Galaxy S2','Grand 2', 'Galaxy Ace']
, "Xiaomi":['Redmi 1s','Redmi 2','Redmi Note 3']}
COMPANY_RATING = {'Apple':5,'Motorola':3,'OnePlus':4,'Samsung':3,'Xiaomi':2}
ISSUE_RATING = {'Hang':1,'None':0,'Battery':2,'Hang+Battery':3,'Microphones':2.5,'Battery+Microphones':4,'Hang+Microphones':3.5,'Hang+Wifi':4.5,'Wifi+Microphones':5}
MODEL_RATING ={'Iphone 4':2,'Iphone 4s':3,'Iphone 5':3.5,'Iphone 5s':4,'Iphone 6':5,
                'G1':2,'G2':3,'G3':4,
                'One':3.5,'Two':4,'X':3,
                'Redmi 1s':2,'Redmi 2':3,'Redmi Note 3':5,
                'Galaxy y':2, 'Galaxy Win':3, 'Galaxy S2':5,'Grand 2':4, 'Galaxy Ace':2,'Galaxy Y':2}

MODEL_FILE = '../models/final_model'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_models/<company>')
def get_models(company):
    print "here "+company
    return json.dumps(PHONES_INFO[company])


@app.route('/predict', methods=['GET','POST'])
def predict():
    data = dict(request.form)
    
    company = int(COMPANY_RATING[data['company'][0]])
    issue   = int(ISSUE_RATING[data['issue'][0]])
    phone_model   = int(MODEL_RATING[data['model'][0]])
    monthsUsed = int(data['months'][0])
    purchase_price = int(data['purchase_price'][0])
    expected_price = int(data['expected_price'][0])
    
    print company
    print issue
    print phone_model
    print monthsUsed
    print purchase_price
    print expected_price
    
    theta = []
    with open(MODEL_FILE,'r') as f:
        model = json.loads(f.read())
        theta = model['theta']
    
    #ORDER OF DATA :company_rating,model_rating,bought_at,months_used,issues_rating,resale_value,output 

	X = [1]
    X.append(company)
    X.append(phone_model)
    X.append(purchase_price)
    X.append(monthsUsed)
    X.append(issue)
    X.append(expected_price)
    
    #Test data that is expected to give output as 1
    #X = [1,4,4,17076,3,3,10137]
    
    z = np.sum(np.multiply(X,theta))
    predictedY = 1/(1 + exp(-z))
    
    if predictedY > 0.5:
    	predictedY = 1
    else:
    	predictedY = 0
    
    print predictedY	
    
    dataList={}
    dataList['company']=company
    dataList['phoneModel']=model
    dataList['issue']=issue
    dataList['expected_price']=expected_price
    dataList['predictedY'] = predictedY
    
    return '<html>PREDICTED' + str(predictedY) + '</html>'
    #return predictedY
    #return render_template('predictedValues.html',dataList=dataList)


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

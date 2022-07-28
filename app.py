import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


application = Flask(__name__)
model = pickle.load(open('house1-price.pkl','rb')) 


@application.route('/')
def home():
  
    return render_template("index.html")
  
@application.route('/predict',methods=['GET'])
def predict():
  r1= float(request.args.get('exp'))
  r2= float(request.args.get('bed'))
  r3= float(request.args.get('bath'))
  r4= float(request.args.get('off'))
  r5= float(request.args.get('brick'))
  r6= float(request.args.get('neigh'))
  
    
  prediction = model.predict([[r1,r2,r3,r4,r5,r6]]) 

    
    
    

    
    
        
  return render_template('index.html', prediction_text='Regression Model  has predicted Price of House for given features is : {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)

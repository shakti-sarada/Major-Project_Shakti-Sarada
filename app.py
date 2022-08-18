import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model1 = pickle.load(open('moviereviewNaiveBayes.pkl','rb'))
model2 = pickle.load(open('moviereviewLogistic.pkl','rb'))
model3 = pickle.load(open('moviereviewdecision_tree.pkl','rb'))
model4 = pickle.load(open('moviereviewrandom.pkl','rb'))
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
print(cv)
corpus=pd.read_csv('corpus_dataset1.csv')
corpus1=corpus['corpus'].tolist()
X = cv.fit_transform(corpus1).toarray()

@app.route('/')
def welcome():
  
    return render_template("home.html")

@app.route('/home')
def home():
  
    return render_template("index.html")
#------------------------------About us-------------------------------------------
@app.route('/aboutme')
def aboutme():
    return render_template('aboutme.html')
  
@app.route('/predict',methods=['GET'])
def predict():
    text = request.args.get('text')
    text=[text]
    input_data = cv.transform(text).toarray()
    
    Model = (request.args.get('Model'))
    if Model=="Naive Bayes Algorithm":
      input_pred = model1.predict(X)
      input_pred = input_pred.astype(int)
      print(input_pred)

    elif Model=="Logistic Regression Algorithm":
      input_pred = model2.predict(X)
      input_pred = input_pred.astype(int)
      print(input_pred)

    elif Model=="Decision Tree Algorithm":
      input_pred = model3.predict(X)
      input_pred = input_pred.astype(int)
      print(input_pred)
      
    else:
      input_pred = model4.predict(X)
      input_pred = input_pred.astype(int)
      print(input_pred)

    if input_pred[0]==1:
        result= "Movie Review is Positive"
    else:
        result="Movie Review is negative" 

    return render_template('index.html', prediction_text='Movie Review Anlaysis: {}'.format(result))

if __name__=="__main__":
    app.run(debug=True)
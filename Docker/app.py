# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:06:54 2022

@author: kiran
"""
import flask
from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import MinMaxScaler
minmax= pickle.load(open("min_max.pkl","rb"))
#minmax = pickle.load('min_max.pkl', 'rb')
assert isinstance(minmax, MinMaxScaler)
minmax.clip = False  # add this line
model =  pickle.load(open("model.pkl","rb"))
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    if request.method == "POST":
        a =  int(request.form["sat"])
        b =  int(request.form["tap"])
        c =  int(request.form["accept"])
        d =  int(request.form["sfratio"])
        e =  int(request.form["expenses"])
        f =  int(request.form["graderate"])
        print(a,b,c,d,e,f)
    result = model.predict(minmax.transform([[a,b,c,d,e,f]])) 
    return render_template("index.html", clust = result)
if __name__ == '__main__':

    app.run(debug = True)
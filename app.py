import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)
global sess

global graph
graph=tf.compat.v1.get_default_graph()

fruitmodel = load_model("D:\\IBM SMARTINTERNZ\\SMARTINTERNZ AI PROJECT NIRMALA\\MODEL\\FRUITS1.h5")
vegmodel=load_model("D:\\IBM SMARTINTERNZ\\SMARTINTERNZ AI PROJECT NIRMALA\\MODEL\\VEGETABLES.h5")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    return render_template('predict.html')

@app.route('/predict',methods=['POST'])		

def predict():
    if request.method == 'POST':

        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(128, 128))
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        plant=request.form['plant']
        print(plant)

        if(plant=="vegetable"):
            preds = vegmodel.predict(x)
            preds = np.argmax(preds)
            print(preds)
            df=pd.read_excel('precautions - veg.xlsx')
            print(df.iloc[preds]['caution'])
        else:
            preds = fruitmodel.predict(x)
            preds = np.argmax(preds)
                
            df=pd.read_excel('precautions - fruits.xlsx')
            print(df.iloc[preds]['caution'])        
        
        return df.iloc[preds]['caution']
     
if __name__ == "__main__":
    app.run(debug=False)

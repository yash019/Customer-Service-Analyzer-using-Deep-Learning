from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow import keras
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app

app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'NLP project\mymodel.h5'

# Load your trained model

#model.save('mymodel.h5')
#model = load_model('mymodel.h5')
#model._make_predict_function()
global graph
graph = tf.get_default_graph()
model = tf.keras.models.load_model('models/mymodel.h5')
#global graph
#graph = tf.get_default_graph()
model._make_predict_function()
#print('Model loaded. Check http://127.0.0.1:5000/')

# dataset = keras.datasets.imdb
# dicti = dataset.get_word_index()

# dicti = {k:(v+3) for (k,v) in dicti.items()}
# dicti["<PAD>"]=0
# dicti["<START>"]=1
# dicti["<UNK>"]=2
# dicti["<UNUSED"]=3

#def decoder(t):
  #text=t.split()
  #decoded_text= [dicti.get(word) for word in text]
  #	return decoded_text

dataset = keras.datasets.imdb
dataset.load_data(num_words=10000)
dicti = dataset.get_word_index()

@app.route('/')
def home():
    return render_template('nlp_project_frontend.html')

@app.route('/predict',methods=['POST'])
def predict():
	from tensorflow import keras


	dataset = keras.datasets.imdb
	dataset.load_data(num_words=10000)
	dicti = dataset.get_word_index()
	#print(dicti)

	dicti = {k:(v+3) for (k,v) in dicti.items()}
	dicti["<PAD>"]=0
	dicti["<START>"]=1
	dicti["<UNK>"]=2
	dicti["<UNUSED"]=3
    #'''
    # For rendering results on HTML GUI
    #'''
	t = request.form['text']
	s=str(t)
	text=s.split()
	decoded_text= [dicti.get(word) for word in text]
	with graph.as_default():
		prediction = model.predict([[decoded_text]])	
	if prediction[[0]]>0.5:
		output="poasitive"
	else:
		output="negative,mind your language"
	return render_template('nlp_project_frontend.html', prediction_text='The input text/speech is $ {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
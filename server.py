from flask import Flask, request, Response
import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import cv2

import keras

from keras.models import load_model

model = load_model("my_model_vgg.h5")
model._make_predict_function()

def preprocess_image(image):
	sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
	return sobely

def sort_key(elem):
	return elem[0]

def predict_classes(path):
	img = cv2.imread(path)
	img = cv2.resize(img, (64,64))
	img = preprocess_image(img)
	img = np.array(img).astype(np.float64)
	img -= np.mean(img)
	img /= np.std(img)
	img = np.expand_dims(img, axis=0)
	prediction = model.predict(img)
	print (prediction)
	arr = []
	for i in range(0, 29):
		arr.append([prediction[0][i], i])
	arr.sort(key=sort_key, reverse=True)
	print (arr)
	print (arr[:3])
	return arr[:3]

# predict_classes

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/GestureImage', methods=['POST'])
def gestureImage():
	r = request
	file = r.files['data'];
	file.save('test.jpg')
	results = predict_classes('test.jpg')
	print (results)
	response = ""
	for result_kv in results:
		result = result_kv[1]
		if (result < 26):
			response += chr(ord('A') + result)
		elif result == 26:
			response += "del"
		elif result == 27:
			response += "nothing"
		else:
			response += "space"
		response += "/"
	return Response(response=response, status=200, mimetype="text")

app.run()
# app.run(host='0.0.0.0')


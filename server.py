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

model = load_model("my_model.h5")
model._make_predict_function()

def predict(path):
	img = cv2.imread(path)
	img = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
	expanded = np.expand_dims(img, axis=0)
	print(model.predict(expanded))

def predict_classes(path):
	img = cv2.imread(path)
	#Samplewise std norm
	#img_std = np.std(np.array(img))
	#img = img - img_std
	#Regular norm
	img = cv2.resize(img, (64,64))
	# img = np.array(img) / 255.0
	# img_std = np.std(np.array(img))
	img = np.array(img).astype(np.float64)
	img -= np.mean(img)
	img /= np.std(img)
	img = np.expand_dims(img, axis=0)
	# my_model = load_model("my_model.h5")
	prediction = model.predict_classes(img)
	return prediction

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/GestureImage', methods=['POST'])
def gestureImage():
	r = request
	file = r.files['data'];
	file.save('test.jpg')
	result = predict_classes('test.jpg')

	print (result)
	response = ""
	if (result < 26):
		response = chr(ord('A') + result)
	elif result == 26:
		response = "del"
	elif result == 27:
		response = "nothing"
	else:
		response = "space"
	return Response(response=response, status=200, mimetype="text")

app.run()
# app.run(host='0.0.0.0')


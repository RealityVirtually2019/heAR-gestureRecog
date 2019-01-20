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

def preprocess_image(image):
	sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
	return sobely

def predict_classes(path):
	img = cv2.imread(path)
	img = cv2.resize(img, (64,64))
	img = preprocess_image(img)
	# img = np.array(img) / 255.0
	# img_std = np.std(np.array(img))
	img = np.array(img).astype(np.float64)
	img -= np.mean(img)
	img /= np.std(img)
	img = np.expand_dims(img, axis=0)
	prediction = model.predict(img)
	print (prediction)
	mnum = 0
	mindex = 0
	for i in range(0, 29):
		if (prediction[0][i] > mnum):
			mnum = prediction[0][i]
			mindex = i
	print (mindex)
	return mindex	


predict_classes('./test/A_test.jpg')
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
	print(model.predict_classes(img))


predict_classes('./test.jpg')
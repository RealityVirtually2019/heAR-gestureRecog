import keras

from keras.models import load_model

model = load_model("my_model.h5")

import numpy as np
import tensorflow as tf
import keras as k

from keras.applications.resnet50 import ResNet50

from keras import backend as K
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import SGD, RMSprop, Adam, Nadam

export_path = './models/'

# Training Not included; We're going to load pretrained weights
# model = load('weights.h5')

# Import the libraries needed for saving models
# Note that in some other tutorials these are framed as coming from tensorflow_serving_api which is no longer correct
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl

# I want the full prediction tensor out, not classification. This format: {"image": Resnet50model.input} took me a while to track down
prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
	{"image": model.input}, 
	{"prediction":model.output})

# export_path is a directory in which the model will be created
builder = saved_model_builder.SavedModelBuilder(export_path)
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

# Initialize global variables and the model
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(init_op)

# Add the meta_graph and the variables to the builder
builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               prediction_signature,
      },
      legacy_init_op=legacy_init_op)
# save the graph      
builder.save()      
import os, random, json
import numpy as np
from scipy import stats
import util

SEQ_SIZE = 8
NUM_TO_GEN = 20
MODEL_DIR = 'trained_all/'
PARSED_DIR = 'parsed_all/'
MAKE_STATEFUL = False
IS_REVERSE = False

#Load titles
title_words, title_word_to_ix = util.load_title_dict(PARSED_DIR)
title_dict_size = len(title_words)
title_sentences = util.load_title_sentences(PARSED_DIR)

#Load comments
comment_words, comment_word_to_ix = util.load_comment_dict(PARSED_DIR)
comment_dict_size = len(comment_words)
comment_sentences = util.load_comment_sentences(PARSED_DIR)
assert(len(title_sentences) == len(comment_sentences))

def word_ixs_to_str(word_ixs, is_title):
	result_txt = ""
	for w_ix in word_ixs:
		w = (title_words if is_title else comment_words)[w_ix]
		if len(result_txt) == 0 or w in ['.', ',', "'", '!', '?', ':', ';', '...']:
			result_txt += w
		elif len(result_txt) > 0 and result_txt[-1] == "'" and w in ['s', 're', 't', 'll', 've', 'd']:
			result_txt += w
		else:
			result_txt += ' ' + w
	if len(result_txt) > 0:
		result_txt = result_txt[:1].upper() + result_txt[1:]
	return result_txt

def probs_to_word_ix(pk, is_first):
	if is_first:
		pk[0] = 0.0
		pk /= np.sum(pk)
	else:
		pk *= pk
		pk /= np.sum(pk)
		#for i in range(3):
		#	max_val = np.amax(pk)
		#	if max_val > 0.5:
		#		break
		#	pk *= pk
		#	pk /= np.sum(pk)

	xk = np.arange(pk.shape[0], dtype=np.int32)
	custm = stats.rv_discrete(name='custm', values=(xk, pk))
	return custm.rvs()

def pred_text(model, context, max_len=64):
	output = []
	context = np.expand_dims(context, axis=0)
	if MAKE_STATEFUL:
		past_sample = np.zeros((1,), dtype=np.int32)
	else:
		past_sample = np.zeros((SEQ_SIZE,), dtype=np.int32)
	while len(output) < max_len:
		pk = model.predict([context, np.expand_dims(past_sample, axis=0)], batch_size=1)[-1]
		if MAKE_STATEFUL:
			pk = pk[0]
		else:
			past_sample = np.roll(past_sample, 1 if IS_REVERSE else -1)
		new_sample = probs_to_word_ix(pk, len(output) == 0)
		past_sample[0 if IS_REVERSE else -1] = new_sample
		if new_sample == 0:
			break
		output.append(new_sample)

	model.reset_states()
	return output

#Load Keras and Theano
print("Loading Keras...")
import os, math
os.environ['KERAS_BACKEND'] = "tensorflow"
import tensorflow as tf
print("Tensorflow Version: " + tf.__version__)
import keras
print("Keras Version: " + keras.__version__)
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, RepeatVector, TimeDistributed, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Convolution1D
from keras.layers.embeddings import Embedding
from keras.layers.local import LocallyConnected2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.models import Model, Sequential, load_model, model_from_json
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1
from keras.utils import plot_model, to_categorical
from keras import backend as K
K.set_image_data_format('channels_first')

#Fix bug with sparse_categorical_accuracy
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
def new_sparse_categorical_accuracy(y_true, y_pred):
	y_pred_rank = ops.convert_to_tensor(y_pred).get_shape().ndims
	y_true_rank = ops.convert_to_tensor(y_true).get_shape().ndims
	# If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
	if (y_true_rank is not None) and (y_pred_rank is not None) and (len(K.int_shape(y_true)) == len(K.int_shape(y_pred))):
		y_true = array_ops.squeeze(y_true, [-1])
	y_pred = math_ops.argmax(y_pred, axis=-1)
	# If the predicted output and actual output types don't match, force cast them
	# to match.
	if K.dtype(y_pred) != K.dtype(y_true):
		y_pred = math_ops.cast(y_pred, K.dtype(y_true))
	return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())

#Load the model
print("Loading Model...")
model = load_model(MODEL_DIR + 'model.h5', custom_objects={'new_sparse_categorical_accuracy':new_sparse_categorical_accuracy})

if MAKE_STATEFUL:
	weights = model.get_weights()
	model_json = json.loads(model.to_json())

	layers = model_json['config']['layers']
	for layer in layers:
		if 'batch_input_shape' in layer['config']:
			layer['config']['batch_input_shape'][0] = 1
			if layer['config']['batch_input_shape'][1] == SEQ_SIZE:
				layer['config']['batch_input_shape'][1] = 1
		if layer['class_name'] == 'Embedding':
			layer['config']['input_length'] = 1
		if layer['class_name'] == 'RepeatVector':
			layer['config']['n'] = 1
		if layer['class_name'] == 'LSTM':
			assert(layer['config']['stateful'] == False)
			layer['config']['stateful'] = True

	print(json.dumps(model_json, indent=4, sort_keys=True))
	model = model_from_json(json.dumps(model_json))
	model.set_weights(weights)

	#plot_model(model, to_file='temp.png', show_shapes=True)

def generate_titles(my_title):
	my_title = util.clean_text(my_title)
	my_words = my_title.split(' ')
	print(' '.join((w.upper() if w in title_word_to_ix else w) for w in my_words) + '\n')
	my_title_ixs = [title_word_to_ix[w] for w in my_words if w in title_word_to_ix]
	my_title_sample = util.bag_of_words(my_title_ixs, title_dict_size)
	for i in range(10):
		print('  ' + word_ixs_to_str(pred_text(model, my_title_sample), False))
	print('')

while True:
	my_title = input('Enter Title:\n')
	generate_titles(my_title)

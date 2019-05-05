import os, sys
import numpy as np
from matplotlib import pyplot as plt
import util

SEQ_SIZE = 8
CTXT_SIZE = 200
EMBEDDING_SIZE = 200
USE_LSTM = True
USE_OUT_SEQ = False
CONTINUE_TRAIN = False
NUM_EPOCHS = 100
NUM_MINI_EPOCHS = 1
BATCH_SIZE = 200
LR = 0.001
DO_RATE = 0.05
BN = 0.99
SAVE_DIR = 'trained_all/'
PARSED_DIR = 'parsed_all/'

#Create directory to save model
if not os.path.exists(SAVE_DIR):
	os.makedirs(SAVE_DIR)

#Load comment dictionary
comment_words, comment_word_to_ix = util.load_comment_dict(PARSED_DIR)
comment_dict_size = len(comment_words)

#Load training samples
title_ix_samples, title_unique_samples, past_samples, pred_samples = util.create_training_samples(PARSED_DIR, SEQ_SIZE, USE_OUT_SEQ)
num_samples = past_samples.shape[0]

#Load Keras and Theano
print("Loading Keras...")
import os, math
#os.environ['THEANORC'] = "./gpu.theanorc"
os.environ['KERAS_BACKEND'] = "tensorflow"
import tensorflow as tf
print("Tensorflow Version: " + tf.__version__)
import keras
print("Keras Version: " + keras.__version__)
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, RepeatVector, TimeDistributed, LeakyReLU, CuDNNGRU, concatenate
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

#Build the training models
if CONTINUE_TRAIN:
	print("Loading Model...")
	model = load_model(SAVE_DIR + 'Model.h5')
else:
	print("Building Model...")
	ctxt_in = Input(shape=title_unique_samples.shape[1:])
	past_in = Input(shape=past_samples.shape[1:])

	if USE_LSTM:
		ctxt_dense = Dense(CTXT_SIZE)(ctxt_in)
		ctxt_dense = LeakyReLU(0.2)(ctxt_dense)
		ctxt_dense = RepeatVector(SEQ_SIZE)(ctxt_dense)

		past_dense = Embedding(comment_dict_size, EMBEDDING_SIZE, input_length=SEQ_SIZE)(past_in)
		x = concatenate([ctxt_dense, past_dense])
		x = Dropout(DO_RATE)(x)

		x = CuDNNGRU(200, return_sequences=USE_OUT_SEQ)(x)
		if USE_OUT_SEQ:
			x = TimeDistributed(BatchNormalization(momentum=BN))(x)
			x = TimeDistributed(Dense(comment_dict_size, activation='softmax'))(x)
		else:
			x = BatchNormalization(momentum=BN)(x)
			x = Dense(comment_dict_size, activation='softmax')(x)
	else:
		ctxt_dense = Dense(CTXT_SIZE)(ctxt_in)
		ctxt_dense = LeakyReLU(0.2)(ctxt_dense)
		past_dense = Embedding(comment_dict_size, EMBEDDING_SIZE, input_length=SEQ_SIZE)(past_in)
		past_dense = Flatten(data_format = 'channels_last')(past_dense)
		x = concatenate([ctxt_dense, past_dense])

		x = Dense(800)(x)
		x = LeakyReLU(0.2)(x)
		if DO_RATE > 0.0:
			x = Dropout(DO_RATE)(x)
		#x = BatchNormalization(momentum=BN)(x)

		x = Dense(400)(x)
		x = LeakyReLU(0.2)(x)
		if DO_RATE > 0.0:
			x = Dropout(DO_RATE)(x)
		#x = BatchNormalization(momentum=BN)(x)

		x = Dense(comment_dict_size, activation='softmax')(x)

	if USE_OUT_SEQ:
		metric = new_sparse_categorical_accuracy
	else:
		metric = 'sparse_categorical_accuracy'

	model = Model(inputs=[ctxt_in, past_in], outputs=[x])
	model.compile(optimizer=Adam(lr=LR), loss='sparse_categorical_crossentropy', metrics=[metric])
	print(model.summary())

	#plot_model(model, to_file=SAVE_DIR + 'model.png', show_shapes=True)

#Utilites
def plotScores(scores, test_scores, fname, on_top=True):
	plt.clf()
	ax = plt.gca()
	ax.yaxis.tick_right()
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.grid(True)
	plt.plot(scores)
	plt.plot(test_scores)
	plt.xlabel('Epoch')
	plt.tight_layout()
	loc = ('upper right' if on_top else 'lower right')
	plt.draw()
	plt.savefig(fname)

#Train model
print("Training...")
train_loss = []
train_acc = []
test_loss = []
test_acc = []
i_train = np.arange(num_samples)
batches_per_epoch = num_samples // BATCH_SIZE
for epoch in range(NUM_EPOCHS):
	np.random.shuffle(i_train)
	for j in range(NUM_MINI_EPOCHS):
		loss = 0.0
		acc = 0.0
		num = 0.0
		start_i = batches_per_epoch * j // NUM_MINI_EPOCHS
		end_i = batches_per_epoch * (j + 1) // NUM_MINI_EPOCHS
		for i in range(start_i, end_i):
			i_batch = i_train[i*BATCH_SIZE:(i + 1)*BATCH_SIZE]
			title_batch = title_unique_samples[title_ix_samples[i_batch]]
			past_batch = past_samples[i_batch]
			pred_batch = pred_samples[i_batch]

			batch_loss, batch_acc = model.train_on_batch([title_batch, past_batch], [pred_batch])
			loss += batch_loss
			acc += batch_acc
			num += 1.0

			if i % 5 == 0:
				progress = ((i - start_i) * 100) // (end_i - start_i)
				sys.stdout.write(
					str(progress) + "%" +
					"  Loss:" + str(loss / num) +
					"  Acc:" + str(acc / num) + "        ")
				sys.stdout.write('\r')
				sys.stdout.flush()
		sys.stdout.write('\n')
		loss /= num
		acc /= num

		train_loss.append(loss)
		train_acc.append(acc)

		plotScores(train_loss, test_loss, SAVE_DIR + 'Loss.png', True)
		plotScores(train_acc, test_acc, SAVE_DIR + 'Acc.png', False)

		if loss == min(train_loss):
			model.save(SAVE_DIR + 'Model.h5')
			print("Saved")

	print("====  EPOCH FINISHED  ====")

print("Done")

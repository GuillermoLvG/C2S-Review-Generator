import sys
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
from tensorflow import keras
import numpy
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import os.path
from category_encoders import BinaryEncoder
from tensorflow.keras.layers import LayerNormalization, Dense, TimeDistributed
from tensorflow.keras.layers import LSTM, Flatten, Embedding, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"
assert tf.__version__ >= "2.0"
import pickle

def get_chars(reviews_path):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = set()
    counter = 0
    with open(reviews_path) as input:
        # lowercase everything to standardize it
        text = input.read().lower()
        # instantiate the tokenizer
        tokens = tokenizer.tokenize(text)
        # if the created token isn't in the stop words, make it part of "filtered"
        filtered = filter(
            lambda token: token not in stopwords.words('english'),
            tokens
        )
    return " ".join(filtered)


# preprocess the input data, make tokens
processed_inputs = get_chars("resources/frankenstein.txt")
chars = sorted(list(set(processed_inputs)))

char_to_num = dict((c, i) for i, c in enumerate(chars))

input_len = len(processed_inputs)
vocab_len = len(chars)
print ("Total number of chars:", input_len)
print ("Total vocab:", vocab_len)


seq_length = 100
x_data = []
y_data = []

# loop through inputs, start at the beginning and go until we hit
# the final character we can create a sequence out of
for i in range(0, input_len - seq_length, 1):
    # Define input and output sequences
    # Input is the current character plus desired sequence length
    in_seq = processed_inputs[i:i + seq_length]

    # Out sequence is the initial character plus total sequence length
    out_seq = processed_inputs[i + seq_length]

    # We now convert list of characters to integers based on
    # previously and add the values to our lists
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

n_patterns = len(x_data)
print ("Total Patterns:", n_patterns)

X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)

y = to_categorical(y_data)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

print(X.shape)
print(y.shape)

model.fit(X, y, epochs=20, batch_size=256, callbacks=desired_callbacks)

# filename = "model_weights_saved.hdf5"
# model.load_weights(filename)
# model.compile(loss='categorical_crossentropy', optimizer='adam')

# num_to_char = dict((i, c) for i, c in enumerate(chars))

# start = numpy.random.randint(0, len(x_data) - 1)
# pattern = x_data[start]
# print("Random Seed:")
# print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

# for i in range(1000):
#     x = numpy.reshape(pattern, (1, len(pattern), 1))
#     x = x / float(vocab_len)
#     prediction = model.predict(x, verbose=0)
#     index = numpy.argmax(prediction)
#     result = num_to_char[index]
#     seq_in = [num_to_char[value] for value in pattern]

#     sys.stdout.write(result)

#     pattern.append(index)
#     pattern = pattern[1:len(pattern)]

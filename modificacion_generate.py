import sys
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
from tensorflow import keras
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import json
from operator import itemgetter
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import os.path
from category_encoders import BinaryEncoder
from tensorflow.keras.layers import LayerNormalization, Dense, TimeDistributed
from tensorflow.keras.layers import LSTM, Flatten, Embedding, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy.sparse as sps
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"
assert tf.__version__ >= "2.0"
import pickle


INITIAL_WORD = "<sos>"
TEST_CONTEXT = 4 #VARIAR NUMERO PARA VER OTROS CONTEXTOS


# SEEDS
np.random.seed(42)
tf.random.set_seed(42)


def check_if_review_stays(review_tokens):
    stopwords_english = stopwords.words('english')
    next_review = False
    if len(review_tokens) > 30:
        next_review = True
    return next_review


def get_contexts_and_reviews():
    print("Getting reviews data")
    decoder_reviews_input = []
    decoder_reviews_target = []
    if os.path.isfile('resources/mod/contexts.pkl'):
        df = pd.read_pickle("resources/mod/contexts.pkl")
        decoder_reviews_input = pickle.load(open("resources/mod/d_input", "rb"))
        decoder_reviews_target = pickle.load(open("resources/mod/d_target", "rb"))
    return encoded_data, decoder_reviews_input, decoder_reviews_target


def create_C2S_model(
    n_features, encoder_embedding_size=25,
        lstm_size=2, MAX_REVIEW_LENGTH=25,
        MAX_VOCAB=10000):
    print("Compiling C2S model")
    # ENCODER
    encoder_input = Input([n_features, ])
    print("Embedding")
    embed = Embedding(n_features, encoder_embedding_size)(encoder_input)
    print("Flatten")
    flattened = Flatten()(embed)
    print("Dense")
    hc = Dense(lstm_size*lstm_size, activation="tanh")(flattened)
    # DECODER
    print("LSTM")
    decoder_lstm = LSTM(
        lstm_size + lstm_size, return_state=True, return_sequences=True)
    decoder_input = Input([MAX_REVIEW_LENGTH,])
    print("Embedding")
    embedded_decoder_input = Embedding(
        MAX_VOCAB, lstm_size)(decoder_input)
    print("Setting initial state")
    decoder_output, _, _ = decoder_lstm(
        embedded_decoder_input, initial_state=[hc, hc])
    print("Dense")
    dense = Dense(MAX_VOCAB, activation='softmax')
    output = dense(decoder_output)
    model = Model(inputs=[encoder_input, decoder_input], outputs=output)
    # model = Model(inputs = encoder_input, outputs = hc)
    model.load_weights("pesos_modificacion.hdf5")
    model.compile(optimizer="adam", loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model.summary()
    return model

def batches_generator(
    contexts, decoder_sequences_input_padded, n=100):
    while True:
        contexts = contexts.tocsr()
        for i in range(contexts.shape[0] // n):
            context = contexts[n*i:n*(i+1)]
            decoder_sequence = decoder_sequences_input_padded[n*i:n*(i+1)]
            yield [context, decoder_sequence]


def add_padding_to_reviews(
    decoder_reviews_input, decoder_reviews_target, MAX_VOCABULARY=10000):
    print("Adding padding to decoder inputs and targets")
    decoder_tokenizer = Tokenizer(num_words=MAX_VOCABULARY, filters='\n')
    decoder_tokenizer.fit_on_texts(
        decoder_reviews_input + decoder_reviews_target)
    decoder_sequences_input = decoder_tokenizer.texts_to_sequences(
        decoder_reviews_input)
    decoder_sequences_target = decoder_tokenizer.texts_to_sequences(
        decoder_reviews_target)
    decoder_index2word = {
        idx: word for word, idx in decoder_tokenizer.word_index.items()
    }
    decoder_sequences_input_padded = pad_sequences(
        decoder_sequences_input, maxlen=25, padding="post")
    decoder_sequences_target_padded = pad_sequences(
        decoder_sequences_target, maxlen=25, padding="post")

    return decoder_sequences_input_padded, decoder_sequences_target_padded,\
        decoder_index2word, decoder_tokenizer.word_index


def main():
    # entonces tengo 1,500,000 reviews
    # los parto en X_train, X_valid, X_test    

    """
    contexts tiene One Hot Encoding de las tuplas unicas. El último valor
    es un flotante, y la variable categorica es el one hot encoding.

    decoder_reviews_input tiene los reviews con <sos> al inicio
    decoder_reviews_target tiene los reviews con <eos> al final
    """
    if os.path.exists("resources/mod/contexts_tr"):
        contexts_tr = pickle.load(open("resources/mod/contexts_tr", "rb"))
        contexts_te = pickle.load(open("resources/mod/contexts_te", "rb"))
        contexts_va = pickle.load(open("resources/mod/contexts_va", "rb"))
        decoder_reviews_input_tr = pickle.load(open(
            "resources/mod/decoder_reviews_input_tr", "rb"))
        decoder_reviews_input_te = pickle.load(open(
            "resources/mod/decoder_reviews_input_te", "rb"))
        decoder_reviews_input_va = pickle.load(open(
            "resources/mod/decoder_reviews_input_va", "rb"))
        decoder_reviews_target_tr = pickle.load(open(
            "resources/mod/decoder_reviews_target_tr", "rb"))
        decoder_reviews_target_te = pickle.load(open(
            "resources/mod/decoder_reviews_target_te", "rb"))
        decoder_reviews_target_va = pickle.load(open(
            "resources/mod/decoder_reviews_target_va", "rb"))
    else:
        contexts_tr, contexts_te, contexts_va,\
            decoder_reviews_input_tr, decoder_reviews_input_te,\
            decoder_reviews_input_va, decoder_reviews_target_tr,\
                decoder_reviews_target_te, decoder_reviews_target_va =\
            get_contexts_and_reviews()
        pickle.dump(contexts_tr, open("resources/mod/contexts_tr", "wb"))
        pickle.dump(contexts_te, open("resources/mod/contexts_te", "wb"))
        pickle.dump(contexts_va, open("resources/mod/contexts_va", "wb"))
        pickle.dump(decoder_reviews_input_tr, open(
            "resources/mod/decoder_reviews_input_tr", "wb"))
        pickle.dump(decoder_reviews_input_te, open(
            "resources/mod/decoder_reviews_input_te", "wb"))
        pickle.dump(decoder_reviews_input_va, open(
            "resources/mod/decoder_reviews_input_va", "wb"))
        pickle.dump(decoder_reviews_target_tr, open(
            "resources/mod/decoder_reviews_target_tr", "wb"))
        pickle.dump(decoder_reviews_target_te, open(
            "resources/mod/decoder_reviews_target_te", "wb"))
        pickle.dump(decoder_reviews_target_va, open(
            "resources/mod/decoder_reviews_target_va", "wb"))

    n_features = contexts_te.shape[1]

    decoder_sequences_input_padded, decoder_sequences_target_padded,\
        decoder_index2word, decoder_word2index = add_padding_to_reviews(
            decoder_reviews_input_te, decoder_reviews_target_te)

    print("Obtaining decoder_targets")
    decoder_targets = np.zeros(
        (
            decoder_sequences_target_padded.shape[0],
            decoder_sequences_target_padded.shape[1],
            10000
        )
    )
    print("Filling decoder_targets")
    for review_idx, review in enumerate(decoder_sequences_target_padded):
        for word_idx, word_id in enumerate(review):
            decoder_targets[review_idx, word_idx, word_id] = 1

    decode_input_value = np.array([[decoder_word2index[INITIAL_WORD]]])
    decode_input_value_padded = pad_sequences(
        decode_input_value, maxlen=25, padding="post")
    n_features = contexts_te.shape[1]
    C2S_model = create_C2S_model(n_features)
    print("Generating embedding")
    # contexts = pickle.load(open("resources/mod/contexts.pkl", "rb"))
    # dec = pickle.load(open("resources/onehotencoder.pkl"))
    # testing_context = dec.inverse_transform(contexts_te[TEST_CONTEXT])
    review = C2S_model.predict(
        [contexts_te[TEST_CONTEXT], decode_input_value_padded])
    text = np.argmax(review[0], axis=1)
    for index in text:
        print(decoder_index2word[index + 1], end=" ")

if __name__ == '__main__':
    main()

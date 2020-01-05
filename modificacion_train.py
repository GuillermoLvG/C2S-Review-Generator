import sys
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
from tensorflow import keras
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import json
from operator import itemgetter
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
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
import random


REVIEW_LENGTH = 25
MAX_VOCAB_LENGTH = 10000
REVIEW_NUMBER = 100000

# SEEDS
np.random.seed(42)
tf.random.set_seed(42)


def text_preprocessor(text):
    """
    Elimina caracteres raros observados en los reviews
    """
    clean = text
    clean = clean.replace("!", "")
    clean = clean.replace("?", "")
    clean = clean.replace("¿", "")
    clean = clean.replace("¡", "")
    clean = clean.replace(".", "")
    clean = clean.replace(",", "")
    clean = clean.replace("'", "")
    clean = clean.replace("-", "")
    clean = clean.replace("_", "")
    clean = clean.replace(")", "")
    clean = clean.replace("(", "")
    clean = clean.replace("$", "")
    clean = clean.replace("/", "")
    clean = clean.replace("\\", "")
    clean = clean.replace("\"", "")
    clean = clean.replace("#", "")
    clean = clean.replace("&", "")
    return clean


def get_contexts_and_reviews():
    """
    Devuelve para train, validation y test (18:1:1)
    los inputs del modelo (contextos, y reviews con <sos>)
    los targets del model (reviews con <eos>)

    Los contextos se duelven codificados con OneHotEncoder

    La generación de esta información sólo ocurre si no existe en la carpeta
    resources/mod/, debido a que toma algo de tiempo generarla.


    """
    print("Getting reviews data")
    decoder_reviews_input = []
    decoder_reviews_target = []
    if os.path.isfile('resources/mod/contexts.pkl'):
        df = pd.read_pickle("resources/mod/contexts.pkl")
        decoder_reviews_input = pickle.load(
            open("resources/mod/d_input", "rb"))
        decoder_reviews_target = pickle.load(
            open("resources/mod/d_target", "rb"))
    else:
        with open("resources/reviews_Movies_and_TV_5.json") as input:
            reviews = input.readlines()
        stop_words = set(stopwords.words('english')) 
        data = []
        random.shuffle(reviews)
        for n, review in enumerate(reviews):
            print(f"Review: {n}")
            review = json.loads(review)
            text = review["reviewText"].lower()
            text = text_preprocessor(text)
            word_tokens = word_tokenize(text)
            filtered_tokens = [w for number, w in enumerate(word_tokens)
                if not w in stop_words and number < REVIEW_LENGTH]
            entry = review['overall']
            decoder_reviews_input.append('<sos> ' + " ".join(filtered_tokens))
            decoder_reviews_target.append(" ".join(filtered_tokens) + ' <eos>')
            data.append(entry)
            if n == REVIEW_NUMBER:
                break;
        print("Encoding Data")
        df = pd.DataFrame(data, columns=['score'])
        df.to_pickle("resources/mod/contexts.pkl")
        pickle.dump(decoder_reviews_input, open(
            "resources/mod/d_input", "wb"))
        pickle.dump(decoder_reviews_target, open(
            "resources/mod/d_target", "wb"))
    categorical_feature_mask = df.dtypes==object
    enc = OneHotEncoder()
    encoded_data = enc.fit_transform(df)
    pickle.dump(enc, open("resources/onehotencoder.pkl", "wb"))
    print(f"# of encoded features {encoded_data.shape[1]}")
    print(encoded_data)
    encoded_data_tr, encoded_data_te, decoder_reviews_input_tr,\
        decoder_reviews_input_te, decoder_reviews_target_tr,\
            decoder_reviews_target_te = train_test_split(
                                                encoded_data,
                                                decoder_reviews_input,
                                                decoder_reviews_target,
                                                test_size=0.1)
    
    encoded_data_tr, encoded_data_va, decoder_reviews_input_tr,\
        decoder_reviews_input_va, decoder_reviews_target_tr,\
            decoder_reviews_target_va = train_test_split(
                                                encoded_data_tr,
                                                decoder_reviews_input_tr,
                                                decoder_reviews_target_tr,
                                                test_size=len(
                                                    decoder_reviews_input_te))
    return encoded_data_tr, encoded_data_te, encoded_data_va,\
        decoder_reviews_input_tr, decoder_reviews_input_te,\
        decoder_reviews_input_va, decoder_reviews_target_tr,\
            decoder_reviews_target_te, decoder_reviews_target_va


def create_C2S_model(
    n_features, batch_size, encoder_embedding_size=25,
        lstm_size=2, MAX_REVIEW_LENGTH=25,
        MAX_VOCAB=10000):
    print("Compiling C2S model")
    #ENCODER
    encoder_input = Input(batch_shape=(batch_size, n_features))
    print("Embedding")
    embed = Embedding(n_features, encoder_embedding_size)(encoder_input)
    print("Flatten")
    flattened = Flatten()(embed)
    print("Dense")
    hc = Dense(lstm_size*lstm_size, activation="tanh")(flattened)
    # #DECODER
    decoder_input = Input(batch_shape=(batch_size, MAX_REVIEW_LENGTH))
    print("Embedding")
    embedded_decoder_input = Embedding(
        MAX_VOCAB, lstm_size)(decoder_input)
    print("Dropout")
    dropped = keras.layers.Dropout(0.2)(embedded_decoder_input)
    print("LSTM")
    decoder_lstm = LSTM(
        lstm_size + lstm_size, return_state=True, return_sequences=True,
            stateful=True)
    print("Setting initial state")
    decoder_output, _, _ = decoder_lstm(dropped, initial_state=[hc, hc])
    dropped = keras.layers.Dropout(0.2)(decoder_output)    
    print("Dense")
    dense = Dense(MAX_VOCAB, activation='softmax')
    output = dense(dropped)
    model = Model(inputs=[encoder_input, decoder_input], outputs=output)
    model.load_weights("pesos_modificacion.hdf5")
    model.compile(optimizer="adam", loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model.summary()
    return model


def batches_generator(
    contexts, decoder_sequences_input_padded, decoder_targets, n=100):
    while True:
        contexts = contexts.tocsr()
        for i in range(contexts.shape[0] // n):
            context = contexts[n*i:n*(i+1)]
            decoder_sequence = decoder_sequences_input_padded[n*i:n*(i+1)]
            target = decoder_targets[n*i:n*(i+1)]
            yield ([context, decoder_sequence], target)


def add_padding_to_reviews(
    decoder_reviews_input, decoder_reviews_target, 
        MAX_VOCABULARY=MAX_VOCAB_LENGTH):
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
        decoder_sequences_input, maxlen=REVIEW_LENGTH, padding="post")
    decoder_sequences_target_padded = pad_sequences(
        decoder_sequences_target, maxlen=REVIEW_LENGTH, padding="post")
    return decoder_sequences_input_padded, decoder_sequences_target_padded,\
        decoder_index2word, decoder_tokenizer.word_index


def main():
    """
    Contexts tiene One Hot Encoding de las tuplas unicas. El último valor
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

    n_features = contexts_tr.shape[1]
    decoder_sequences_input_padded, decoder_sequences_target_padded,\
        decoder_index2word, decoder_word2index = add_padding_to_reviews(
            decoder_reviews_input_tr, decoder_reviews_target_tr)

    decoder_sequences_input_padded_va, decoder_sequences_target_padded_va,\
        decoder_index2word_va, decoder_word2index_va = add_padding_to_reviews(
            decoder_reviews_input_va, decoder_reviews_target_va)

    print("Obtaining decoder_targets")
    decoder_targets = np.zeros(
        (
            decoder_sequences_target_padded.shape[0],
            decoder_sequences_target_padded.shape[1],
            MAX_VOCAB_LENGTH
        ), dtype=bool
    )
    print("Filling decoder_targets")
    for review_idx, review in enumerate(decoder_sequences_target_padded):
        for word_idx, word_id in enumerate(review):
            decoder_targets[review_idx, word_idx, word_id] = 1

    print("Obtaining decoder_targets_va")
    decoder_targets_va = np.zeros(
        (
            decoder_sequences_target_padded_va.shape[0],
            decoder_sequences_target_padded_va.shape[1],
            MAX_VOCAB_LENGTH
        ), dtype=bool
    )
    print("Filling decoder_targets_va")
    for review_idx, review in enumerate(decoder_sequences_target_padded_va):
        for word_idx, word_id in enumerate(review):
            decoder_targets_va[review_idx, word_idx, word_id] = 1
    batch_size = 100  # Batch para generar los embeddings.
    print(contexts_tr.shape)
    C2S_model = create_C2S_model(n_features, batch_size)
    steps = contexts_tr.shape[0] // batch_size
    steps_for_va = contexts_va.shape[0] // batch_size

    gen = batches_generator(
        contexts_tr, decoder_sequences_input_padded,
            decoder_targets, batch_size)

    validation_gen = batches_generator(
        contexts_va, decoder_sequences_input_padded_va, decoder_targets_va
    )

    filepath = "pesos_modificacion.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    desired_callbacks = [checkpoint]
    r = C2S_model.fit_generator(
            gen,
            steps_per_epoch=steps,
            epochs=20,
            verbose=1,
            callbacks=desired_callbacks,
            validation_data=validation_gen,
            validation_steps=steps_for_va
        )

if __name__ == '__main__':
    main()

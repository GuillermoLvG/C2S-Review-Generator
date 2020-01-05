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
import scipy.sparse as sps
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"
assert tf.__version__ >= "2.0"
import pickle


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
    if os.path.isfile('resources/contexts.pkl'):
        df = pd.read_pickle("resources/contexts.pkl")
        decoder_reviews_input = pickle.load(open("resources/d_input", "rb"))
        decoder_reviews_target = pickle.load(open("resources/d_target", "rb"))
    else:
        with open("resources/reviews_Movies_and_TV_5.json") as input:
            reviews = input.readlines()
        data = []
        tokenizer = RegexpTokenizer(r'\w+')
        for n, review in enumerate(reviews):
            print(f"Review: {n}")
            review = json.loads(review)
            text = review["reviewText"].lower()
            review_tokens = tokenizer.tokenize(text)
            next_review = check_if_review_stays(review_tokens)
            if next_review:
                continue
            entry = (review['asin'], review['overall'])
            if entry not in data:
                decoder_reviews_input.append('<sos> ' + text)
                decoder_reviews_target.append(text + ' <eos>')
                data.append(entry)
            else:
                continue
        print("Encoding Data")
        df = pd.DataFrame(data, columns = ['product', 'score'])
        df.to_pickle("resources/contexts.pkl")
        pickle.dump(decoder_reviews_input, open("resources/d_input", "wb"))
        pickle.dump(decoder_reviews_target, open("resources/d_target", "wb"))
    # aquí se hace el encoding del dataframe
    # data = np.array(df)
    # data = data.to_dict(orient='records')
    # enc = DictVectorizer()
    # encoded_data = enc.fit_transform(data)
    # enc = BinaryEncoder()
    # encoded_data = enc.fit_transform(data)
    # Categorical boolean mask
    categorical_feature_mask = df.dtypes==object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = df.columns[categorical_feature_mask].tolist()
    le = LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
    enc = OneHotEncoder(categorical_features=categorical_feature_mask)
    encoded_data = enc.fit_transform(df)
    print(f"# of encoded features {encoded_data.shape[1]}")
    # print(f"Example: {encoded_data.toarray()}")
    return encoded_data, decoder_reviews_input, decoder_reviews_target


def create_C2S_model(
    n_features, sos_embedding, encoder_embedding_size=100,
        lstm_size=2, MAX_REVIEW_LENGTH=100,
        MAX_VOCAB=20000):
    print("Compiling C2S model")
    #ENCODER
    encoder_input = Input([n_features, ])
    print("Embedding")
    embed = Embedding(n_features, encoder_embedding_size)(encoder_input)
    print("Flatten")
    flattened = Flatten()(embed)
    print("Dense")
    hc = Dense(lstm_size*lstm_size, activation="tanh")(flattened)
    # #DECODER
    decoder_input = Input([MAX_REVIEW_LENGTH, ])
    print("Embedding")
    embedded_decoder_input = Embedding(
        MAX_VOCAB, lstm_size)(decoder_input)
    sos_embedding = embedded_decoder_input[0][1]
    print("LSTM")
    decoder_lstm = LSTM(
        lstm_size + lstm_size, return_state=True, return_sequences=True)
    print("Setting initial state")
    # print(hc)
    # print(type(hc))
    # print(sos_embedding)
    # print(type(sos_embedding))
    decoder_output, _, _ = decoder_lstm(
        embedded_decoder_input, initial_state=[hc, hc])
    print("Dense")
    dense = Dense(MAX_VOCAB, activation='softmax')
    output = dense(decoder_output)
    model = Model(inputs=[encoder_input, decoder_input], outputs=output)
    # model = Model(inputs = encoder_input, outputs = hc)
    model.compile(optimizer="adam", loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model.summary()
    return model

# pagina 310

# Use Concatenate para juntar input con output de cierto punto. Es decir
# es para poner los skip conections

# def create_C2S_model(encoder, decoder):
#     model = keras.models.Sequential([stacked_encoder, stacked_decoder])
#     model.compile(
#         loss="binary_crossentropy",
#         optimizer=keras.optimizers.SGD(lr=1.5),
#         metrics=[rounded_accuracy]
#     )    
#     return model


def batches_generator(
    contexts, decoder_sequences_input_padded, decoder_targets, n=100):
    while True:
        contexts = contexts.tocsr()
        print(len(decoder_sequences_input_padded))
        print(decoder_targets.shape)
        for i in range(contexts.shape[0] // n):
            context = contexts[n*i:n*(i+1)]
            decoder_sequence = decoder_sequences_input_padded[n*i:n*(i+1)]
            target = decoder_targets[n*i:n*(i+1)]
            yield ([context, decoder_sequence], target)


def add_padding_to_reviews(
    decoder_reviews_input, decoder_reviews_target, MAX_VOCABULARY=20000):
    print("Adding padding to decoder inputs and targets")
    decoder_tokenizer = Tokenizer(num_words=MAX_VOCABULARY, filters = '\n')
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
        decoder_sequences_input, maxlen = 30, padding = "post")
    decoder_sequences_target_padded = pad_sequences(
        decoder_sequences_target, maxlen = 30, padding = "post")
    return decoder_sequences_input_padded, decoder_sequences_target_padded,\
        decoder_index2word, decoder_tokenizer.word_index


class NDSparseMatrix:
    def __init__(self):
        self.elements = {}

    def __getitem__(self, item):
        if isinstance(item, slice):
            print("SLICE")
            print(item)
        return self.elements[item]

    def set_shape(self):
        self.shape = self.get_shape()

    def get_shape(self):
        tuples = (self.elements.keys())
        shape = (
            max(tuples, key=itemgetter(1))[0],
            max(tuples, key=itemgetter(1))[1],
            max(tuples, key=itemgetter(1))[2]
        )
        return shape
    
    def addValue(self, tuple, value):
        self.elements[tuple] = value

    def readValue(self, tuple):
        try:
            value = self.elements[tuple]
        except KeyError:
            # could also be 0.0 if using floats...
            value = False
        return value


def main():
    # entonces tengo 1,500,000 reviews
    # los parto en X_train, X_valid, X_test    

    """
    contexts tiene One Hot Encoding de las tuplas unicas. El último valor
    es un flotante, y la variable categorica es el one hot encoding.

    decoder_reviews_input tiene los reviews con <sos> al inicio
    decoder_reviews_target tiene los reviews con <eos> al final
    """
    contexts, decoder_reviews_input, decoder_reviews_target =\
        get_contexts_and_reviews()
    n_features = contexts.shape[1]
    decoder_sequences_input_padded, decoder_sequences_target_padded,\
        decoder_index2word, decoder_word2index = add_padding_to_reviews(
            decoder_reviews_input, decoder_reviews_target)

    print("Obtaining decoder_targets")
    decoder_targets = np.zeros(
        (
            decoder_sequences_target_padded.shape[0],
            5000,
            decoder_sequences_target_padded.shape[1]
        ), dtype=bool
    )
    print("Filling decoder_targets")
    for review_idx, review in enumerate(decoder_sequences_target_padded):
        for word_idx, word_id in enumerate(review):
            decoder_targets[review_idx, word_idx, word_id] = 1

    # decoder_targets = NDSparseMatrix()
    # print("Filling decoder_targets")
    # for review_idx, review in enumerate(decoder_sequences_target_padded):
    #     for word_idx, word_id in enumerate(review):
    #         tuple = (review_idx, word_idx, word_id)
    #         value = True
    #         decoder_targets.addValue(tuple, value)
    # decoder_targets.set_shape()

    sos_embedding = decoder_word2index['<sos>']
    C2S_model = create_C2S_model(n_features, sos_embedding)
    
    batch_size = 32  # Batch para generar los embeddings.
    steps = contexts.shape[0] // batch_size
    gen = batches_generator(
        contexts, decoder_sequences_input_padded, decoder_targets, batch_size)

    r = C2S_model.fit_generator(
        gen, steps_per_epoch=steps, epochs=1, verbose=1)

    # EL QUE CESAR Y YO QUEDAMOS QUE FUNCIONA
    # r = C2S_model.fit(x = [contexts, decoder_sequences_input_padded],
    #       y = decoder_targets, 
    #       epochs = 20, 
    #       batch_size= 128, validation_set = x_valid)


    # C2S_model = create_C2S_model(n_features)

    # batch_size = 32  # Batch para generar los embeddings.
    # steps = contexts.shape[0] // batch_size
    # context_generator = batches_generator(contexts, batch_size)
    # print("Generating embedding")
    # context_embeddings = C2S_model.predict_generator(
    #     context_generator,
    #     steps = steps,
    #     max_queue_size=2,  # Para usar menos memoria
    #     verbose = 1)
    # print(context_embeddings)
    # print(context_embeddings.shape)
    # embeddings = C2S_model.predict(contexts.toarray(), batch_size=1, steps=10)
    # print(embeddings.shape)

if __name__ == '__main__':
    main()

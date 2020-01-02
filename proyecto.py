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
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import os.path
from category_encoders import BinaryEncoder
from tensorflow.keras.layers import LayerNormalization, Dense, TimeDistributed
from tensorflow.keras.layers import LSTM, Flatten, Embedding
from tensorflow.keras.models import Sequential, Model
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"
assert tf.__version__ >= "2.0"
import pickle


# SEEDS
np.random.seed(42)
tf.random.set_seed(42)

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


def get_top_20k_words(reviews):
    tokenizer = RegexpTokenizer(r'\w+')
    word_freq = {}
    stopwords_english = stopwords.words('english')
    for n, review in enumerate(reviews):
        print(f"Review: {n}")
        review = json.loads(review)
        text = review["reviewText"].lower()
        tokens = tokenizer.tokenize(text)
        for word in tokens:
            if word in stopwords_english:
                continue
            if word in word_freq:
                word_freq[word] = word_freq[word] + 1
            else:
                word_freq[word] = 1
    top_20k = sorted(word_freq.keys())[:20000]
    with open('resources/top_20k', 'wb') as output:
        pickle.dump(top_20k, output)
    with open('resources/tokens', 'wb') as output:
        pickle.dump(list(word_freq.keys()), output)
    return top_20k, tokens


def check_if_review_stays(review_tokens, top_20k):
    stopwords_english = stopwords.words('english')
    next_review = False
    for word in review_tokens:
        if word in stopwords_english:
            continue
        if word not in top_20k:
            next_review = True
            break
    if len(review_tokens) > 100:
        next_review = True
    return next_review

def get_contexts():
    print("Getting reviews data")
    if os.path.isfile('resources/contexts.pkl'):
        df = pd.read_pickle("resources/contexts.pkl")
    else:
        with open("resources/reviews_Movies_and_TV_5.json") as input:
            reviews = input.readlines()
        if os.path.isfile('resources/top_20k'):
            with open('resources/top_20k', 'rb') as input:
                top_20k = pickle.load(input)
            with open('resources/tokens', 'rb') as input:
                tokens = pickle.load(input)
        else:
            top_20k, tokens = get_top_20k_words(reviews)
        tokenizer = RegexpTokenizer(r'\w+')
        for n, review in enumerate(reviews):
            print(f"Review: {n}")
            review = json.loads(review)
            data = set()
            text = review["reviewText"].lower()
            review_tokens = tokenizer.tokenize(text)
            next_review = check_if_review_stays(review_tokens, top_20k)
            if next_review:
                next_review = False
                continue
            with open("reviews_text.txt", "a+") as output:
                output.write(text + "\n")
            entry = (review['asin'], review['overall'])
            data.add(entry)
        print("Generating Dataframe")
        df = pd.DataFrame(data, columns = ['product', 'score'])
        df.to_pickle("resources/contexts.pkl")
    return df


def encode_contexts(data):
    print("Encoding data")
    data = np.array(data)

    # data = data.to_dict(orient='records')
    # enc = DictVectorizer()
    # encoded_data = enc.fit_transform(data)

    enc = BinaryEncoder()
    encoded_data = enc.fit_transform(data)

    # enc = OneHotEncoder()
    # encoded_data = enc.fit_transform(data)
    print(f"# of encoded features {encoded_data.shape[1]}")
    return encoded_data


input_shape=(X.shape[1], X.shape[2])

def create_context_decoder_model(hc, n_words, states_length, batches_size):
    print("Compiling context encoder")
    model = Sequential()
    model.add(LSTM(
            256,
            return_sequences=True,
            stateful=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            batch_input_shape = [batches_size, n_words, states_length]
        )
    )
    model.add(TimeDistributed(Dense(states_length, activation="softmax")))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam"
    )
    model.layers[0].states[0] = hc
    model.summary()
    return model


def create_context_encoding_model(n_features, n_contexts, embedding_size=100):
    print("Compiling context encoder")
    model = Sequential()
    model.add(Embedding(n_features, embedding_size, input_length=n_features))
    model.add(Flatten())
    model.add(Dense(64, activation="tanh"))
    model.compile('rmsprop', 'mse')
    model.summary()
    return model


def batches_generator(contexts, n=100):
    while True:
        for i in range(contexts.shape[0] // n):
            value = contexts[n*i:n*(i+1)]
            yield value


class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()


def generate_context_embeddings(contexts):
    encoded_contexts = encode_contexts(contexts)
    n_contexts = encoded_contexts.shape[0]
    n_features = encoded_contexts.shape[1]
    embedding_model = create_context_encoding_model(n_features, n_contexts)
    # batch_size = contexts.shape[0]  # Batch para generar los embeddings.
    # steps = contexts.shape[0] // batch_size
    # context_generator = batches_generator(encoded_contexts, batch_size)
    # print("Generating embedding")
    # context_embeddings = embedding_model.predict_generator(
    #     context_generator,
    #     steps = steps,
    #     max_queue_size=2,  # Para usar menos memoria
    #     verbose = 1)
    # return context_embeddings
    return embedding_model

# pagina 310

# Use Concatenate para juntar input con output de cierto punto. Es decir
# es para poner los skip conections

def create_C2S_model(encoder, decoder):
    model = keras.models.Sequential([stacked_encoder, stacked_decoder])
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.SGD(lr=1.5),
        metrics=[rounded_accuracy]
    )    
    return model

def main():
    contexts = get_contexts()
    encoder = generate_context_embeddings(contexts)
    # context_embeddings = generate_context_embeddings(contexts)
    # n_contexts = context_embeddings.shape[0]
    # n_features = context_embeddings.shape[1]
    n_contexts = len(contexts)
    n_features = 16
    decoder = create_context_decoder_model(
        context_embeddings,
        n_features,
        200,
        n_contexts
    )
    C2S = create_C2S_model(encoder, decoder)
    history = C2S.fit(X_train, X_train, epochs=20,
                         validation_data=[X_valid, X_valid])

if __name__ == '__main__':
    main()

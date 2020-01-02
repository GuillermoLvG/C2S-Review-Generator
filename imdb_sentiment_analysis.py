import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
import numpy as np
import tensorflow_datasets as tfds
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter

# SEEDS
np.random.seed(42)
tf.random.set_seed(42)

# pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "nlp"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def get_dataset_info():
    datasets, info = tfds.load(
        "imdb_reviews",
        as_supervised=True,
        with_info=True
    )
    return datasets, info


def get_sizes(info):
    train_size = info.splits["train"].num_examples
    test_size = info.splits["test"].num_examples
    return train_size, test_size


def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch


def get_words_and_word_ids(vocab_size, datasets):
    vocabulary = Counter()
    for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
        for review in X_batch:
            vocabulary.update(list(review.numpy()))
    truncated_vocabulary = [
        word for word, count in vocabulary.most_common()[:vocab_size]]
    word_to_id = {word: index for index, word in enumerate(truncated_vocabulary)}
    words = tf.constant(truncated_vocabulary)
    word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
    return words, word_ids


def get_train_set(num_oov_buckets, vocab_init, datasets):

    def encode_words(X_batch, y_batch):
        return table.lookup(X_batch), y_batch

    num_oov_buckets = 1000
    table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)
    train_set = datasets["train"].repeat().batch(32).map(preprocess)
    train_set = train_set.map(encode_words).prefetch(1)
    return train_set




def main_process():
    datasets, info = get_dataset_info()
    train_size, test_size = get_sizes(info)
    vocab_size = 10000
    words, word_ids = get_words_and_word_ids(vocab_size, datasets)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    num_oov_buckets = 1000    
    train_set = get_train_set(num_oov_buckets, vocab_init, datasets)
    embed_size = 128
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                            mask_zero=True,  # not shown in the book
                            input_shape=[None]),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.GRU(128),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    history = model.fit(train_set, steps_per_epoch=train_size // 32, epochs=5)


main_process()
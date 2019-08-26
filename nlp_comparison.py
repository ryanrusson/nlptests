"""
NOTES:
    Different models to can be used to create 'word vectors' (think numerical vector space, but for text)
    - Word2vec algorithm (most popular): one implementation in gensim python lib
    - GLoVe (Global Vectors): no python lib implementation; similar to matrix factorization;
        here's a link to the Stanford site:  https://nlp.stanford.edu/projects/glove/
    - FastText (Facebook created):  use "pip install fasttext" to install
    For visualization, look at t-SNE, in sklearn.manifold ("TSNE")
"""

# Import in some stuff
import re

import pandas as pd
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

### (START) Extra stuff to make my RTX card work ###
# https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)
### (END) Extra stuff to make my RTX card work ###

# Function for cleaning up the review text
def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)

    # Strip escaped quotes
    text = text.replace('\\"', '')

    # Strip quotes
    text = text.replace('"', '')

    return text


# Encoder for the binary response of the 'positive' and 'negative' reviews
def bin_encoder(y):
    """
    Function to take an pd.Series input with 'positive' and 'negative'
    :param y: a pd.Series with 'positive' and 'negative' in the vector
    :return: a pd.Series with 1 or 0 for the values
    """
    if y.lower() == 'positive':
        return 1
    elif y.lower() == 'negative':
        return 0
    else:
        raise ValueError("ERR0R! There is an unexpected value in your response data!")


# Control Logic (for faster testing)
deepnn = False
cnn = False
lstm = True
transfer = False

# Reading in the review text from IMDB Sentiment dataset
df = pd.read_csv('MovieReviewTrainingDatabase.csv')
df['cleaned_review'] = df['review'].apply(clean_review)
df['encode_sentiment'] = df['sentiment'].apply(bin_encoder)
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['encode_sentiment'], test_size=0.2)


# Build the count vectorizer
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=5000)
X_train_onehot = vectorizer.fit_transform(X_train)

### Using a Dense Network for Modeling ###
if deepnn:
    # Build the simple NN
    model = Sequential()
    model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    print("Training on a Dense NN...")
    model.fit(X_train_onehot[:-500], y_train[:-500], epochs=3, batch_size=128, verbose=1,
              validation_data=(X_train_onehot[-500:], y_train[-500:]))


    # Now evaluate the data on the test set
    scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
    print("Accuracy:", scores[1])


### Create word vectors of the data ###
word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
tokenize = vectorizer.build_tokenizer()
preprocess = vectorizer.build_preprocessor()


def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes


print(to_sequence(tokenize, preprocess, word2idx, "This is an important test!"))
X_train_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_train]
print(X_train_sequences[0])

# PROBLEM! The sequences are of different lengths. We solve this by padding the to the left with 5000.
# Compute the max length of a text
MAX_SEQ_LENGHT = len(max(X_train_sequences, key=len))
print("MAX_SEQ_LENGHT=", MAX_SEQ_LENGHT)

N_FEATURES = len(vectorizer.get_feature_names())
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGHT, value=N_FEATURES)
print(X_train_sequences[0])

X_test_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_test]
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGHT, value=N_FEATURES)


### Using Convolutional Neural Network for Modeling ###
if cnn:
    model = Sequential()
    model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                        64,  # Embedding size
                        input_length=MAX_SEQ_LENGHT))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    print("Training on a CNN...")
    model.fit(X_train_sequences[:-500], y_train[:-500],
              epochs=3, batch_size=512, verbose=1,
              validation_data=(X_train_sequences[-500:], y_train[-500:]))

    scores = model.evaluate(X_test_sequences, y_test, verbose=1)
    print("Accuracy:", scores[1])


### Using LSTM for Modeling ###
if lstm:
    model = Sequential()
    model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                        64,  # Embedding size
                        input_length=MAX_SEQ_LENGHT))
    model.add(LSTM(64))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    print("Training on a LSTM...")
    model.fit(X_train_sequences[:-100], y_train[:-100],
              epochs=2, batch_size=512, verbose=1,
              validation_data=(X_train_sequences[-100:], y_train[-100:]))

    scores = model.evaluate(X_test_sequences, y_test, verbose=1)
    print("Accuracy:", scores[1])

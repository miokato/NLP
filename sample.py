import numpy as np
from gensim.models import word2vec
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Reshape, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from preprocess.loader import Loader
from preprocess.vectorization import train_test_split
import settings


MAXLEN = 60
VOCAB_SIZE = 50
HIDDEN_SIZE = 256
BATCH_SIZE = 128


inputs = Loader.load(settings.INPUTS)
outputs = Loader.load(settings.OUTPUTS)

embeddings_model = word2vec.Word2Vec.load('models/char2vec.model')

input_char_list = [[c for c in sent] for sent in inputs]
output_char_list = [[c for c in sent] for sent in outputs]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_char_list)
tokenizer.fit_on_texts(output_char_list)
input_seq = tokenizer.texts_to_sequences(input_char_list)
output_seq = tokenizer.texts_to_sequences(output_char_list)
x = pad_sequences(input_seq, maxlen=MAXLEN)
y = pad_sequences(output_seq, maxlen=MAXLEN)
(x_train, x_test), (y_train, y_test) = train_test_split(x, y, test_percent=10)

word_index = tokenizer.word_index
num_char = len(word_index)

embedding_matrix = np.zeros((num_char+1, VOCAB_SIZE))
for char, i in word_index.items():
    if char in embeddings_model.wv.index2word:
        embedding_matrix[i] = embeddings_model[char]


if __name__ == '__main__':
    model = Sequential()
    model.add(Embedding(
        input_dim=num_char + 1,
        output_dim=VOCAB_SIZE,
        input_length=MAXLEN,
        weights=[embedding_matrix],
        trainable=False
    ))
    model.add(LSTM(HIDDEN_SIZE))
    model.add(Dense(MAXLEN, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.summary()
    early_stopping = EarlyStopping()
    model.fit(x_train, y_train, epochs=10, callbacks=[early_stopping])
    model.save(settings.MODEL)


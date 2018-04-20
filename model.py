from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape
from keras.layers.wrappers import TimeDistributed


def model_1(input_length, output_length, vocab_size, hidden_size=256):
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(input_length, vocab_size)))
    model.add(Dense(output_length * vocab_size))
    model.add(Reshape((output_length, vocab_size)))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    return model




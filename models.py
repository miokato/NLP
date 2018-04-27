from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape
from keras.layers.wrappers import TimeDistributed
from seq2seq.models import SimpleSeq2Seq


def model_1(input_length, output_length, vocab_size, hidden_size=256):
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(input_length, vocab_size)))
    model.add(Dense(output_length * vocab_size))
    model.add(Reshape((output_length, vocab_size)))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    return model


def model_2(input_length, output_length, vocab_size, hidden_size=256):
    model = Sequential()
    model.add(SimpleSeq2Seq(
        input_dim=vocab_size,
        input_length=input_length,
        output_length=output_length,
        output_dim=vocab_size,
        hidden_dim=hidden_size,
    ))
    model.add(Dense(output_length * vocab_size))
    model.add(Reshape((output_length, vocab_size)))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    return model


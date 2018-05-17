from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Reshape
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


def seq2seq(encoder_vocab, decoder_vocab, hidden_dim):

    encoder_inputs = Input(shape=(None, encoder_vocab))
    encoder = LSTM(hidden_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, decoder_vocab))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(decoder_vocab, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


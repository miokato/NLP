import pickle

from samples.addition_rnn import Generator


with open('data/c_to_i.pkl', 'rb') as f:
    char_index = pickle.load(f)
generator = Generator(path='models/addition_rnn.h5', vocab=char_index, maxlen=50)
print(generator.generate('タコメシ？'))

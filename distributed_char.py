import logging
from gensim.models import word2vec


def text2char():
    with open ('data/ja.text8') as f:
        text = f.read()

    chars = [c for c in text if c != ' ']

    with open('data/ja.char8', 'w') as f:
        f.write(' '.join(chars))


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('data/ja.char8')
model = word2vec.Word2Vec(sentences, size=50, window=50, workers=4)
model.save('models/char2vec.model')



from preprocess import Loader, create_corpus, convert_onehot

l = []
path = 'data/inputs.txt'
loader = Loader()
data = loader.load(path)


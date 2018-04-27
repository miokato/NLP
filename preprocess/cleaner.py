from preprocess.loader import Loader
import settings


def text_cleaner():
    sents = []
    data = Loader.load(settings.RAW_DATA)
    for sent in data:
        if sent != '\n':
            sents.append(sent)
    Loader.save(sents, settings.CLEANED)


if __name__ == '__main__':
    text_cleaner()

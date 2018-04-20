import pickle
import numpy as np
import settings


def get_char_id() -> dict:
    return pickle.load(open(settings.CHAR_ID_PKL, 'rb'))


def get_id_char() -> dict:
    return pickle.load(open(settings.ID_CHAR_PKL, 'rb'))


def get_vocab_size() -> int:
    return len(get_char_id())


def encode(sentence: str, input_length: int) -> np.ndarray:
    char_id = get_char_id()
    vocab_size = get_vocab_size()
    x = np.zeros((input_length, vocab_size))
    for i, char in enumerate(sentence):
        if not char == ' ':
            x[i, char_id[char]] = 1
    return x


def decode(x: np.ndarray, calc_argmax=True) -> str:
    id_char = get_id_char()
    if calc_argmax:
        x = x.argmax(axis=-1)
    return ''.join(id_char[i] for i in x)



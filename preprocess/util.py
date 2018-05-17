import numpy as np


class WordTable:

    def __init__(self, bow: str or list):
        self.bow = sorted(set(bow))
        self.char_indices = dict((c, i) for i, c in enumerate(self.bow))
        self.indices_char = dict((i, c) for i, c in enumerate(self.bow))

    def encode(self, sent: str, num_rows: int) -> np.ndarray:
        x = np.zeros((num_rows, len(self.bow)))
        for i, c in enumerate(sent):
            x[i, self.char_indices[c]] = 1.
        return x

    def decode(self, x: np.ndarray, calc_argmax=True) -> str:
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[i] for i in x)


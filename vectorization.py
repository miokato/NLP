import numpy as np

import settings
from loader import Loader
from corpus import get_vocab_size, encode


def vectorization():
    queries = Loader.load(settings.INPUTS)
    answers = Loader.load(settings.OUTPUTS)

    # 1つ目の長さでいいのか?
    input_max_len = len(queries[0])
    output_max_len = len(answers[0])
    vocab_size = get_vocab_size()

    x = np.zeros((len(queries), input_max_len, vocab_size), dtype=np.bool)
    y = np.zeros((len(queries), output_max_len, vocab_size), dtype=np.bool)

    # encode queries
    for i, sent in enumerate(queries):
        print(i)
        sent = sent.strip('\n')
        x[i] = encode(sent, input_max_len)
    # encode answers
    for i, sent in enumerate(answers):
        print(i)
        sent = sent.strip('\n')
        y[i] = encode(sent, output_max_len)

    x, y = randomize(x, y, len(y))

    (x_train, x_test), (y_train, y_test) = train_test_split(x, y, test_percent=10)

    np.savez_compressed(
        settings.XY_VECTORS_NPZ,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test
    )


def randomize(x: np.ndarray, y: np.ndarray, size: int) -> [np.ndarray, np.ndarray]:
    indices = np.arange(size)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    return x, y


def train_test_split(x: np.ndarray, y: np.ndarray, test_percent: int):
    split_at = len(x) - (len(x) // test_percent)
    (x_train, x_test) = x[:split_at], x[split_at:]
    (y_train, y_test) = y[:split_at], y[split_at:]

    return (x_train, x_test), (y_train, y_test)




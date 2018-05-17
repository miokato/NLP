from keras.models import Sequential, load_model
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np


class CharacterTable:

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[idx] for idx in x)


class Generator:

    def __init__(self, path, vocab='0123456789+ ', maxlen=7):
        self.path = path
        self.maxlen = maxlen
        self.vocab = vocab
        self.table = CharacterTable(self.vocab)
        self.model = load_model(self.path)

    def generate(self, mes):
        query = mes + ' ' * (self.maxlen - len(mes))  # padding
        query = query[::-1]  # 逆順の方が精度が出るらしい
        vec = self.table.encode(query, self.maxlen)
        vec = np.array([vec])  # reshape -> (1, x, y)
        a_vec = self.model.predict_classes(vec)
        answer = self.table.decode(a_vec[0], calc_argmax=False)
        return answer


def seq2seq(maxlen, vocab_size, out_len=3):
    """model"""
    RNN = layers.LSTM
    HIDDEN_SIZE = 128
    LAYERS = 1

    print('Build model..')
    model = Sequential()
    model.add(RNN(HIDDEN_SIZE, input_shape=(maxlen, vocab_size)))
    model.add(layers.RepeatVector(out_len + 1))

    for _ in range(LAYERS):
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(vocab_size)))
    model.add(layers.Activation('softmax'))

    return model


def corpus_gen(maxlen, vocab, reverse=True):
    ctable = CharacterTable(vocab)
    questions = []
    expected = []
    seen = set()
    TRAINING_SIZE = 50000
    DIGITS = 3

    print('Generating data...')
    while len(questions) < TRAINING_SIZE:
        f = lambda : int(''.join(np.random.choice(list('0123456789'))
                                 for i in range(np.random.randint(1, DIGITS + 1))))
        a, b = f(), f()
        key = tuple(sorted((a, b)))  # 昇順に並び替えて、以後重複していたら生成しない
        if key in seen:
            continue
        seen.add(key)

        q = '{}+{}'.format(a, b)
        query = q + ' ' * (maxlen - len(q))  # padding

        ans = str(a + b)
        ans += ' ' * (DIGITS + 1 - len(ans))  # padding

        if reverse:
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))

    print('Vectorization')
    x = np.zeros((len(questions), maxlen, len(vocab)), dtype=np.bool)
    y = np.zeros((len(questions), DIGITS + 1, len(vocab)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, maxlen)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, DIGITS + 1)

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    return (x_train, x_val), (y_train, y_val)


def train():
    """
    maxlen (int) : timestamp 文字列の場合、センテンスの長さ
    vocab (list) : 文字および単語のリスト
    reverse (bool) : reverseするかどうか

    :return:
    """
    DIGITS = 3  # 数値の最大長 100の位まで指定可能
    MAXLEN = DIGITS + 1 + DIGITS  # inputの最大長 questionはxxx+yyyなので(例 123+456)
    REVERSE = True

    chars = '0123456789+ '
    ctable = CharacterTable(chars)

    BATCH_SIZE = 128

    # data
    (x_train, x_val), (y_train, y_val) = corpus_gen(maxlen=MAXLEN, vocab=chars, reverse=True)
    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)

    # callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    check_point = ModelCheckpoint(filepath='models/addition_rnn.h5', monitor='val_loss',
                                  verbose=1, save_best_only=True, mode='auto')

    # model
    model = seq2seq(maxlen=MAXLEN, vocab_size=len(chars))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.summary()
    model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, check_point]
    )

    # check
    for i in range(10):
        idx = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([idx])], y_val[np.array([idx])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print('OK', end=' ')
        else:
            print('NG', end=' ')
        print(guess)


if __name__ == '__main__':
    train()

from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pickle

from samples.addition_rnn import CharacterTable, seq2seq


with open('data/corpus.distinct.txt', 'rt') as f:
    data = f.readlines()

with open('data/c_to_i.pkl', 'rb') as f:
    char_index = pickle.load(f)

vocab_size = len(char_index)
REVERSE = True
maxlen = 50
batch_size = 128
# training_size = 100000
chars = list(char_index.keys())
table = CharacterTable(chars)

questions, answers = [], []
for sent in data:
    sent = sent.strip()
    question, answer = sent.split('__SP__')
    if len(question) > 50:
        question = question[:50]
    else:
        question = question + ' ' * (maxlen - len(question))
    question = question[::-1]
    questions.append(question)

    if len(answer) > 50:
        answer = answer[:50]
    else:
        answer = answer + ' ' * (maxlen + 1 - len(answer))
    answers.append(answer)

x = np.zeros((len(questions), maxlen, vocab_size), dtype=np.bool)
y = np.zeros((len(answers), maxlen + 1, vocab_size), dtype=np.bool)
for i, sent in enumerate(questions):
    x[i] = table.encode(sent, maxlen)
for i, sent in enumerate(answers):
    y[i] = table.encode(sent, maxlen + 1)

indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]


early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
check_point = ModelCheckpoint(filepath='models/addition_rnn.h5', monitor='val_loss',
                              verbose=1, save_best_only=True, mode='auto')
# model
model = seq2seq(maxlen=maxlen, vocab_size=len(chars), out_len=maxlen)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()
model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=500,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, check_point]
)

# check
for i in range(10):
    idx = np.random.randint(0, len(x_val))
    rowx, rowy = x_val[np.array([idx])], y_val[np.array([idx])]
    preds = model.predict_classes(rowx, verbose=0)
    q = table.decode(rowx[0])
    correct = table.decode(rowy[0])
    guess = table.decode(preds[0], calc_argmax=False)
    print('Q', q[::-1] if REVERSE else q, end=' ')
    print('T', correct, end=' ')
    if correct == guess:
        print('OK', end=' ')
    else:
        print('NG', end=' ')
    print(guess)

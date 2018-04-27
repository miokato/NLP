import numpy as np

from keras.callbacks import EarlyStopping
import settings
from model import model_1
from preprocess.corpus import decode


print('Loading data')
data = np.load(settings.XY_VECTORS_NPZ)
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

INPUT_LENGTH = x_train.shape[1]
OUTPUT_LENGTH = y_train.shape[1]
VOCAB_SIZE = x_train.shape[2]

print(x_train.shape)
print(y_train.shape)

HIDDEN_SIZE = 256
BATCH_SIZE = 256

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model = model_1(INPUT_LENGTH, OUTPUT_LENGTH, VOCAB_SIZE, hidden_size=HIDDEN_SIZE)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

for _ in range(1, 100):
    print()
    print('-' * 50)

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=1,
        validation_data=(x_test, y_test)
    )
    for i in range(10):
        idx = np.random.randint(0, len(x_test))
        row_x, row_y = x_test[np.array([idx])], y_test[np.array([idx])]
        preds = model.predict_classes(row_x, verbose=0)
        q = decode(row_x[0])
        correct = decode(row_y[0])
        guess = decode(preds[0], calc_argmax=False)
        print('Q', q)
        print('T', correct)
        if correct == guess:
            print('ok')
        else:
            print('fail')
        print(guess)
        print('---')
model.save(settings.MODEL)
print('Finish training')

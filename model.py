from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(monitor='val_loss', patience=2)




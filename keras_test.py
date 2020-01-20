import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Lambda, Dense

timesteps = 1
x = pd.read_csv('pre_processed_data/data_x.csv', sep=';')
x = np.reshape(np.array(x), (x.shape[0], timesteps, x.shape[1]))
y = pd.read_csv('pre_processed_data/data_y.csv', sep=';')


N_INPUTS = 286
N_FEATURES = 70
N_BLOCKS = 286
N_OUTPUTS = 13
model = Sequential()

model.add(LSTM(N_BLOCKS, input_shape=(N_INPUTS, N_FEATURES)))
model.add(Dense(N_OUTPUTS))
# model.add(LSTM(1, input_shape=(timesteps, data_dim), return_sequences=True))
# model.add(Lambda(lambda x: x[:, -N:, :]))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # fit the keras model on the dataset
model.fit(x, y, epochs=150, batch_size=10)
#
# _, accuracy = model.evaluate(x, y)
# print('Accuracy: %.2f' % (accuracy * 100))

from keras.layers import Dense, Flatten, LSTM, Activation
from keras.layers import Dropout, RepeatVector, TimeDistributed
from keras import Input, Model

from keras.layers import Dense, Flatten, LSTM, Activation
from keras.layers import Dropout, RepeatVector, TimeDistributed
from keras import Input, Model

seq_length = 15
input_dims = 46
output_dims = 2 # number of classes
n_hidden = 10
model1_inputs = Input(shape=(seq_length,input_dims,))
model1_outputs = Input(shape=(output_dims,))

net1 = LSTM(n_hidden, return_sequences=True)(model1_inputs)
net1 = LSTM(n_hidden, return_sequences=False)(net1)
net1 = Dense(output_dims, activation='relu')(net1)
model1_outputs = net1

model1 = Model(inputs=model1_inputs, outputs=model1_outputs, name='model1')


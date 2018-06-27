import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from env import Guy

g = Guy()
g.reset()

use_dropout = True

model = Sequential()
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))


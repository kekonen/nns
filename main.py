# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation, TimeDistributed, GRU, LSTM
from keras.optimizers import Adam
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier

alphabet = 'abcdefghijklmnopqrstuvwxyz'

EPISODES = 400

sets = 500

wordsToRemember = ['abcd','aefg','aejk','aejc']

def OneHot2letter(oneHot):
    for i, val in enumerate(oneHot):
        if val == 1:
            return alphabet[i]
    return 'a'

def letter2OneHot(letter):
    oneHotLetter = np.zeros( len(alphabet))
    oneHotLetter[alphabet.index(letter)] = 1
    return oneHotLetter

def arrayOfWords2OneHot(dataset, shuffled=True):
    newDataset = []
    for word in dataset:
        oneHotWordx = [] #np.zeros((len(word)-1, len(alphabet)))
        oneHotWordy = [] #np.zeros((len(word)-1, len(alphabet)))
        for i in range(3):
            # oneHotWordx[i][alphabet.index(letter)] = 1
            oneHotWordx.append(letter2OneHot(word[i]))
            oneHotWordy.append(letter2OneHot(word[i+1]))
        newDataset.append([oneHotWordx, oneHotWordy])
    if shuffled: random.shuffle(newDataset)
    xDst = []
    yDst = []
    for i in newDataset:
        xDst.append(i[0])
        yDst.append(i[1])
    return np.array(xDst), np.array(yDst)

def generateDataset(times):
    datax = []
    datay = []
    for i in range(times):
        x,y = arrayOfWords2OneHot(wordsToRemember)
        datax.append(x)
        datay.append(y)
    return np.concatenate(datax), np.concatenate(datay)


# lol = arrayOfWords2OneHot(wordsToRemember)

x_train, y_train = generateDataset(sets)
print(x_train.shape, y_train.shape)


def vanilla_rnn():
    model = Sequential()
    # model.add(SimpleRNN(50, input_shape = (None, 3, 26), return_sequences = False))
    # model.add(Dense(26))
    # model.add(Activation('softmax'))
    # adam = Adam(lr = 0.001)
    #model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    model.add(LSTM(input_dim  =  26, output_dim = 50, return_sequences = True))
    model.add(TimeDistributed(Dense(output_dim = 26, activation  =  "softmax")))
    model.compile(loss = "categorical_crossentropy", optimizer = "adam")
    
    
    return model

model = vanilla_rnn()



# model = KerasClassifier(build_fn = vanilla_rnn, epochs = 200, batch_size = 50, verbose = 1)

model.fit(x_train, y_train, epochs=10, batch_size=50, verbose=2)

model.save_weights('save/vanillarnn.h5')

first =  letter2OneHot('a').reshape(1,26)
second = letter2OneHot('e').reshape(1,26)
third =  letter2OneHot('f').reshape(1,26)

toPredict = np.array([first,second,third]).reshape(1,3,26)
print('lol->',toPredict)

prediction = model.predict(toPredict)
print(prediction)

for i in prediction[0]:
    print(alphabet[np.argmax(i)])
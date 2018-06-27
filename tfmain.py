from env import Guy
import numpy as np
# import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation
# from keras.layers import LSTM

g = Guy()
g.reset()
o, [r, s, t], score = g.act('up') # observation, [featurable_reward, featurable_score, time_passed], score



use_dropout = True
n_epoch = 1

games_run = 1
n_batch   = min_step_length = 1 # g.min_steps
n_frames  = 12
n_inputs  = 491
n_neurons = 512
think_n_steps = 10

def make_step(observation, to_go, r, s, t):
	return np.concatenate((observation.flatten(), np.array([r, s, t]), to_go),axis=0).reshape((1, n_inputs))

def to_go_from(lol):
	z =np.zeros(4)
	z[g.board_flat.index(potential)] = 1
	return z





# model = Sequential()
# model.add(LSTM(64, input_shape=(n_frames, n_inputs), return_sequences=True))
# model.add(LSTM(64, return_sequences=True))
# if use_dropout:
#     model.add(Dropout(0.5))
# model.add(TimeDistributed(Dense(vocabulary)))
# model.add(Activation('tanh'))

# X = seq.reshape(len(seq), 1, 1)
# y = seq.reshape(len(seq), 1)
# # define LSTM configuration
# n_neurons = length
# n_batch = length
# n_epoch = 1000
# # create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(n_inputs, n_frames)))
model.add(Dense(1))
model.add(Activation('tanh'))

model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
# model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
# result = model.predict(X, batch_size=n_batch, verbose=0)

for game_i in range(games_run):
	g.reset()

	potential = ['up','left','right','down'][np.random.randint(4)]
	to_go = to_go_from(potential)

	o, [r, s, t], score = g.act(potential)

	X = np.concatenate((o.flatten(), np.array([r, s, t]), to_go),axis=0).reshape((491,1))
	y = np.array(r).reshape((1,1))

	# while not score:
	for i in range(n_frames-1):
		maxx = 0
		potential = ['up','left','right','down'][np.random.randint(4)]
		for direction in g.ways_to_go():
			to_go = to_go_from(potential)
			final_result = model.predict(make_step(o, to_go, r, s, t), batch_size=n_batch, verbose=0)
			final_result = 0
			# for i in range(think_n_steps):
			# 	result += model.predict(make_step(observation, to_go), batch_size=n_batch, verbose=0)
			if final_result > maxx:
				maxx = final_result
				potential = direction
		o, [r, s, t], score = g.act(potential)
		gone =  to_go_from(potential)
		# if i == 0:
		# 	X = np.stack((X, np.concatenate((o.flatten(), np.array([r, s, t]), gone),axis=0)), axis=0 )
		# 	y = np.stack((y, np.array(r).reshape((1,))), axis=0)
		# else:
		X = np.concatenate((X, np.concatenate((o.flatten(), np.array([r, s, t]), gone),axis=0).reshape(n_inputs, 1)), axis=1 )
		y = np.concatenate((y, np.array(r).reshape((1,1))), axis=0)

	# model.fit(X, y, epochs=n_epoch, batch_size=1, verbose=2)


		


	




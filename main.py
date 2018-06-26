import numpy as np
from random import randrange
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import keras


with open('data.train', 'r') as f:
	raw = f.read()

raw = raw.split('\n')



train_features = np.zeros((145,4))
train_labels   = np.zeros((145,3))

test_features = np.zeros((5,4))
test_labels   = np.zeros((5,3))

max0 = 7.9
min0 = 4.3
max1 = 4.4
min1 = 2.2
max2 = 6.9
min2 = 1.0
max3 = 2.5
min3 = 0.1

# index_left = [x for x in range(dataset_len)]

def dividee(xx):
	xx[0] = (xx[0] - min0) / (max0 - min0)
	xx[1] = (xx[1] - min1) / (max1 - min1)
	xx[2] = (xx[2] - min2) / (max2 - min2)
	xx[3] = (xx[3] - min3) / (max3 - min3)
	return xx

def shuffle(raw):
	rawnew = raw[:]
	train_features = np.zeros((145,4))
	train_labels   = np.zeros((145,3))
	test_features = np.zeros((5,4))
	test_labels   = np.zeros((5,3))
	dataset_len  = 150
	dataset_left = 150
	shuffled = []
	for i in range(dataset_len):
		n = randrange(dataset_left)
		shuffled.append(rawnew[n])
		del rawnew[n]
		dataset_left-=1
	for i in range(len(shuffled)):
		splitted = shuffled[i].split(',')
		for ii in range(4):
			# print(splitted[ii])
			splitted[ii] = float(splitted[ii])
		for ii in range(3):
			splitted[ii+4] = int(splitted[ii+4])
		# print(i, splitted)
		if i<145:
			train_features[i] = dividee(splitted[:4])
			train_labels[i]   = splitted[4:]
		else:
			# print('test',i,)
			test_features[i-145]  = dividee(splitted[:4])
			test_labels[i-145]    = splitted[4:]
			# print( splitted, test_features,test_labels)
	return train_features, train_labels, test_features, test_labels

train_features, train_labels, test_features, test_labels = shuffle(raw)

n_features  = 4
n_classes   = 3

batch_size  = 145
n_nodes_hl1 = 10
n_nodes_hl2 = 10

hm_epochs   = 900


############################################################################################
# model = Sequential()


# model.add(Dense(units=10, activation='relu', input_dim=4))
# model.add(Dense(units=10, activation='relu'))
# model.add(Dense(units=3, activation='softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])

# # model.fit(x_train, y_train, epochs=5, batch_size=1)

# for i in range(hm_epochs):
# 	train_features, train_labels, test_features, test_labels = shuffle(raw)
# 	model.train_on_batch(train_features, train_labels)

# 	loss_and_metrics = model.evaluate(test_features, test_labels)
# 	print(loss_and_metrics)

# print(test_features)
# print(test_labels)
# classes = model.predict(test_features)
# print(classes)


# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))


################################################################################################

x = tf.placeholder('float', [None, n_features])
y = tf.placeholder('float', [None, n_classes ])
# x_train = tf.placeholder('float', [5, n_features])
# y_train = tf.placeholder('float', [5, n_classes ])
keep_prob = tf.placeholder(tf.float32)


def neural_network_model(data, keep_prob):

	hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([n_features, n_nodes_hl1], stddev=0.1)),
		'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2])), 
		'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	# hidden_3_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3])), 
	# 	'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_classes])), 
		'biases':tf.Variable(tf.random_normal([n_classes]))}


	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	
	l1_drop = tf.nn.dropout(l1, keep_prob)

	l2 = tf.add(tf.matmul(l1_drop, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2) 

	# l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	# l2 = tf.nn.relu(l2)
	output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])
	output = tf.nn.softmax(output) 


	return output


def train_model(x, keep_prob):
	prediction = neural_network_model(x, keep_prob)
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
	train_step = tf.train.AdamOptimizer(5e-3).minimize(cross_entropy)

	# correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
	# accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	# print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


	with tf.Session() as sess:
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			train_features, train_labels, test_features, test_labels = shuffle(raw)
			print('starting epoch:',epoch, 'of', hm_epochs)
			for i in range(11):
				if i%5 == 0:
					train_accuracy = accuracy.eval(feed_dict={x:test_features, y: test_labels, keep_prob: 0.9})
					print("step %d, training accuracy %g"%(i, train_accuracy))
				train_step.run(feed_dict={x: train_features, y: train_labels, keep_prob: 0.9})

				# print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

		print(train_features[0].reshape(1,4), train_labels[0].reshape(1,3))
		print(sess.run(cross_entropy, feed_dict={x: train_features[0].reshape(1,4), y: train_labels[0].reshape(1,3), keep_prob: 0.6}))
		print(sess.run(prediction, feed_dict={x: train_features[0].reshape(1,4), y: train_labels[0].reshape(1,3), keep_prob: 0.6}))
		


train_model(x, keep_prob)

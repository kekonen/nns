import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data', one_hot=True)

n_nodes_cnv1 = 32
n_nodes_cnv2 = 64

n_nodes_hl1 = 1024
# n_nodes_hl2 = 500
# n_nodes_hl3 = 500

n_classes   = 10
batch_size  = 100
hm_epochs   = 10

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10 ])
keep_prob = tf.placeholder(tf.float32)


def neural_network_model(data, keep_prob):
	conv_1_layer = {'weights':tf.Variable(tf.truncated_normal([5, 5, 1, n_nodes_cnv1], stddev=0.1)), 
		'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_cnv1]))}

	conv_2_layer = {'weights':tf.Variable(tf.truncated_normal([5, 5, 32, n_nodes_cnv2], stddev=0.1)), 
		'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_cnv2]))}

	hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([7 * 7 * n_nodes_cnv2, n_nodes_hl1], stddev=0.1)),
		'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	# hidden_2_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2])), 
	# 	'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	# hidden_3_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3])), 
	# 	'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_classes])), 
		'biases':tf.Variable(tf.random_normal([n_classes]))}


	x_image = tf.reshape(data, [-1,28,28,1])

	h_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(x_image, conv_1_layer['weights'], strides=[1, 1, 1, 1], padding='SAME'), conv_1_layer['biases']))
	h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	h_conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(h_pool1, conv_2_layer['weights'], strides=[1, 1, 1, 1], padding='SAME'), conv_2_layer['biases']))
	h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*n_nodes_cnv2])

	l1 = tf.add(tf.matmul(h_pool2_flat, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	
	l1_drop = tf.nn.dropout(l1, keep_prob)

	# l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	# l2 = tf.nn.relu(l2)

	# l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	# l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l1_drop, output_layer['weights']), output_layer['biases'])

	return output



def train_model(x, keep_prob):
	prediction = neural_network_model(x, keep_prob)
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	# correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
	# accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	# print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


	with tf.Session() as sess:
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		sess.run(tf.global_variables_initializer())

		for i in range(20000):
			batch = mnist.train.next_batch(50)
			if i%100 == 0:
				train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
				print("step %d, training accuracy %g"%(i, train_accuracy))
			train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

			print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

		


train_model(x, keep_prob)

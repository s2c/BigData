from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 200000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.75, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')

#keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


def train():
	# Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True,fake_data=FLAGS.fake_data)

	sess = tf.InteractiveSession()

	# Create a multilayer model.

	# Input placehoolders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 784], name='x-input')
		image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
		tf.image_summary('input', image_shaped_input, 10)
		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
		keep_prob = tf.placeholder(tf.float32)
		tf.scalar_summary('dropout_keep_probability', keep_prob)

	# function to create a convolutional layer
	def createConvLayer(inputImage, theWeights, theBiases):
		return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inputImage, theWeights, strides=[1, 1, 1, 1], padding='SAME'),theBiases))	

	# function to create a max pool layer
	def createMaxPoolLayer(inputImage, kernelSize):
		return tf.nn.max_pool(inputImage, ksize=[1, kernelSize, kernelSize, 1], strides=[1, kernelSize, kernelSize, 1], padding='SAME')			
	
	# function to create the desired network
	def createConvNetwork(theInput, dropoutKeepProb):
		# The input image is stored as vector 784 unit vector. Reshape it into a square 28x28x1 image (1 for only 1 color channel, grayscale)
		theInput = tf.reshape(theInput, shape=[-1, 28, 28, 1])

		############ 1st Convolutional Block ########################
		# the convolution
		layerWeights = tf.Variable(tf.random_normal([5, 5, 1, 32])) #5x5 kernel size, 1 input depth to 32 input depths
		layerBiases = tf.Variable(tf.random_normal([32]))
		conv1 = createConvLayer(theInput, layerWeights, layerBiases)
		# the max pooling
		conv1 = createMaxPoolLayer(conv1, kernelSize=2)
		# the dropout
		conv1 = tf.nn.dropout(conv1, dropoutKeepProb) 

		############ 2nd Convolutional Block ########################
		# the convolution
		layerWeights = tf.Variable(tf.random_normal([5, 5, 32, 64])) #5x5 kernel, 32 depths to 64 depths
		layerBiases = tf.Variable(tf.random_normal([64]))
		conv2 = createConvLayer(conv1, layerWeights, layerBiases)
		# the max pooling
		conv2 = createMaxPoolLayer(conv2, kernelSize=2)
		# the dropout
		conv2 = tf.nn.dropout(conv2, dropoutKeepProb) 

		##################### Fully connected layer ########################
		layerWeights = tf.Variable(tf.random_normal([3136, 1024])) # 3136 inputs (from 7*7*64) , 1024 outputs
		layerBiases = tf.Variable(tf.random_normal([1024]))
		dense1 = tf.reshape(conv2, [-1, layerWeights.get_shape().as_list()[0]]) # Reshape output of 2nd convolutional block to fit the input of the dense layer
		# the RELU
		dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, layerWeights), layerBiases))
		# the dropout
		dense1 = tf.nn.dropout(dense1, dropoutKeepProb) 

		###################### Final "layer", comparing classes ############################
		layerWeights = tf.Variable(tf.random_normal([1024, 10])) #1024 inputs and 10 outputs (1 for each class)
		layerBiases = tf.Variable(tf.random_normal([10])) 
		out = tf.add(tf.matmul(dense1, layerWeights), layerBiases)
		return out

	def variable_summaries(var, name):
		"""Attach a lot of summaries to a Tensor."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.scalar_summary('mean/' + name, mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
			tf.scalar_summary('sttdev/' + name, stddev)
			tf.scalar_summary('max/' + name, tf.reduce_max(var))
			tf.scalar_summary('min/' + name, tf.reduce_min(var))
			tf.histogram_summary(name, var)

 
	#Actually Create the neural network model
	y = createConvNetwork(x, keep_prob) #these are the predictions, ie: the output of the network is predictions
	# Define loss and optimizer
#	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
#	optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)
	
	
	with tf.name_scope('cross_entropy'):
		with tf.name_scope('total'):
			# Define loss function
			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
		tf.scalar_summary('cross entropy', cross_entropy)

	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.scalar_summary('accuracy', accuracy)

	# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
	merged = tf.merge_all_summaries()
	train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
	test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')


	# Train the model, and also write summaries.
	# Every 100th step, measure test-set accuracy, and write test summaries
	# All other steps, run train_step on training data, & add training summaries
#	tf.initialize_all_variables().run()
	sess.run(tf.initialize_all_variables())
	for i in range(FLAGS.max_steps):
		xs, ys = mnist.train.next_batch(128, fake_data=FLAGS.fake_data)
		k = FLAGS.dropout
		#fit the training datra for this batch
		sess.run(train_step, feed_dict={x:xs, y_:ys, keep_prob:k})
		if i % 1000 ==0: #record summaries and test-set accuracy
			xs, ys = mnist.test.images, mnist.test.labels
			k = 1.0
			summary, acc = sess.run([merged, accuracy], feed_dict={x: xs, y_: ys, keep_prob: k})
			test_writer.add_summary(summary, i)
			print('Accuracy at step %s: %s' % (i, acc))


def main(_):
	if tf.gfile.Exists(FLAGS.summaries_dir):
		tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
	tf.gfile.MakeDirs(FLAGS.summaries_dir)
	train()

if __name__ == '__main__':
	tf.app.run()

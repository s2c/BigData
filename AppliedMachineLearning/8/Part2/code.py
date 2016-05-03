from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 50000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 50, 'Number of examples in a batch.')
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs.')
flags.DEFINE_integer('test_stride', 1000, 'Test accuracy is recorded every x steps')
flags.DEFINE_float('init_learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
#flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('data_dir', 'MNIST_data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')



def train():
	#Load MNIST Data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

	sess = tf.InteractiveSession()


	#---------------------------------------Create the model------------------------------------

	############################
	# Placeholders and Variables
	############################
	with tf.name_scope('input'):
		#start building the computation graph by creating nodes for the input images and target output classes
		x = tf.placeholder(tf.float32, shape=[None, 784])
		y_ = tf.placeholder(tf.float32, shape=[None, 10])

		#create a placeholder for the probability that a neuron's output is kept during dropout
		keep_prob = tf.placeholder(tf.float32)
		#create a placeholder for the adaptive learning rate
		learning_rate = tf.placeholder(tf.float32)


	##################################################################
	# ------------------CONVOLUTIONAL NEURAL NET----------------------
	##################################################################

	#######################
	# Weight Initialization
	#######################
	#using ReLU neurons, so initialize with a slightly positive initial bias to avoid "dead neurons" 
	#Instead of doing this repeatedly while we build the model, create two handy functions to do it for us
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)


	#########################
	# Convolution and Pooling
	#########################

	#vanilla version: 
	#-convolutions use a stride of 1 and are zero padded so the output is the same size as the input
	#-pooling is plain old max pooling over 2x2 blocks

	#abstract those operations into functions.
	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


	###########################
	# First Convolutional Layer
	###########################
	# First layer will consist of convolution, followed by max pooling
	# convolutional will compute 8 features for each 5x5 patch 
		#Its weight tensor will have a shape of [5, 5, 1, 8]. The first two dimensions are the patch size, 
		#the next is the number of input channels, and the last is the number of output channels. 
	W_conv1 = weight_variable([5, 5, 1, 8])
	# We will also have a bias vector with a component for each output channel
	b_conv1 = bias_variable([8])

	#To apply the layer, first reshape x to a 4d tensor, with the second and third dimensions corresponding 
	#to image width and height, and the final dimension corresponding to the number of color channels.
	x_image = tf.reshape(x, [-1,28,28,1])

	#convolve x_image with the weight tensor, add the bias, apply the ReLU function
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	#dropout
	h_conv1 = tf.nn.dropout(h_conv1, keep_prob)

	############################
	# Second Convolutional Layer
	############################
	#To build a deep network, stack several layers of this type. 
	#The second layer will have 8 features as well for each 5x5 patch
	W_conv2 = weight_variable([5, 5, 8, 8])
	b_conv2 = bias_variable([8])
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
	#dropout
	h_conv2 = tf.nn.dropout(h_conv2, keep_prob)

	############################
	# Third Convolutional Layer
	############################
	#To build a deep network, stack several layers of this type. 
	#This layer will have 8 features as well for each 5x5 patch
	W_conv3 = weight_variable([5, 5, 8, 8])
	b_conv3 = bias_variable([8])
	h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
	#dropout
	h_conv3 = tf.nn.dropout(h_conv3, keep_prob)

	############################
	# Third Convolutional Layer
	############################
	#To build a deep network, stack several layers of this type. 
	#This layer will have 8 features as well for each 5x5 patch
	W_conv4 = weight_variable([5, 5, 8, 8])
	b_conv4 = bias_variable([8])
	h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
	#dropout
	h_conv4 = tf.nn.dropout(h_conv4, keep_prob)

	#########################
	# Densely Connected Layer
	#########################
	# image size is still 28x28. we add a fully-connected layer with 1024 neurons to allow processing on the entire image
	W_fc1 = weight_variable([28 * 28 * 8, 1024])
	b_fc1 = bias_variable([1024])

	#reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
	flat = tf.reshape(h_conv4, [-1, 28*28*8])
	h_fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)

	#########
	# Dropout
	#########
	#To reduce overfitting, we will apply dropout before the readout layer
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	###############
	# Readout Layer
	###############
	#Finally, we add a softmax layer, just like for the one layer softmax regression from the simple tutorial.
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)






	#--------------------------------------- TRAINING ----------------------------------------
	##############################
	# Train and Evaluate the Model
	##############################

	#use code that is nearly identical to that for the simple one layer SoftMax network above
	#differences are: 
		#replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer
		#include the additional parameter keep_prob in feed_dict to control the dropout rate
		#log every 100th iteration in the training process

	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
		tf.scalar_summary('cross entropy', cross_entropy)
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.scalar_summary('accuracy', accuracy)

	merged = tf.merge_all_summaries()
	train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
	test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

	tf.initialize_all_variables().run()	
	#sess.run(tf.initialize_all_variables())
	
	
	def feed_dict(train, currLR):
		#Make a TensorFlow feed_dict: maps data onto Tensor placeholders.
		if train or FLAGS.fake_data:
			xs, ys = mnist.train.next_batch(FLAGS.batch_size, fake_data=FLAGS.fake_data)
			k = FLAGS.dropout
		else:
			xs, ys = mnist.test.images, mnist.test.labels
			k = 1.0
		return {x: xs, y_: ys, keep_prob: k, learning_rate: currLR}
	
	lr = FLAGS.init_learning_rate #initialize the learning rate... it will be decreased each epoch
	for i in range(FLAGS.max_steps):
		if i % FLAGS.test_stride == 0:  # Record summaries and test-set accuracy
		      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False, lr))
		      test_writer.add_summary(summary, i)
		      print('------------------------------Test Accuracy at step %s: %s' % (i, acc))
		else: # Record train set summaries, and train
			summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True, lr))
			train_writer.add_summary(summary, i)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict=feed_dict(True, lr))
				print("step %d, training accuracy %g"%(i, train_accuracy))

		#split the training range into epochs... each epoch decrease the learning rate
		if i% (FLAGS.max_steps / FLAGS.num_epochs) == 0:  
			lr = lr - 3.6e-7

#		batch = mnist.train.next_batch(FLAGS.batch_size)
#		#measure the accuracy
#		if i%100 == 0:
#			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
#			#train_writer.add_summary(summary, i)
#			print("step %d, training accuracy %g"%(i, train_accuracy))		
#
#		#train the model on the current batch
#		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: FLAGS.dropout, learning_rate: lr})
#	
#		#split the training range into 200 (50000/250) epochs... each epoch decrease the learning rate
#		if i% (FLAGS.max_steps / FLAGS.num_epochs) == 0:  
#			lr = lr - 3.6e-7

	print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def main(_):
	if tf.gfile.Exists(FLAGS.summaries_dir):
		tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
	tf.gfile.MakeDirs(FLAGS.summaries_dir)
	train()

if __name__ == '__main__':
  tf.app.run()















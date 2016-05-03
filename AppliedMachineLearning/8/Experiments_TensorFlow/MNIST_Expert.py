#Load MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


import tensorflow as tf
sess = tf.InteractiveSession()


############################
# Placeholders and Variables
############################
#start building the computation graph by creating nodes for the input images and target output classes
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#define the weights W and biases b for our model
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


#Before Variables can be used within a session, they must be initialized using that session. 
#The following line takes the initial values (in this case tensors full of zeros) that have 
#already been specified, and assigns them to each Variable. 
#This can be done for all Variables at once.
sess.run(tf.initialize_all_variables())


###################################
# Predicted Class and Cost Function
###################################

# regression model
y = tf.nn.softmax(tf.matmul(x,W) + b)

# cost function will be the cross-entropy between the target and the model's prediction
#note: tf.reduce_sum sums across all classes and tf.reduce_mean takes the average over these sums
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


#################
# Train the Model
#################
# use steepest gradient descent, with a step length of 0.5, to descend the cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#The returned operation train_step, when run, will apply the gradient descent updates to the parameters

#Training the model can therefore be accomplished by repeatedly running train_step.
#Each training iteration we...
#	load 50 training examples
#	run the train_step operation, using feed_dict to replace the placeholder tensors x and y_ with the training examples.
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})


####################
# Evaluate the Model
####################
#tf.argmax gives you the index of the highest entry in a tensor along some axis
#tf.argmax(y,1) is the label our model thinks is most likely for each input
#tf.argmax(y_,1) is the correct label
#use tf.equal to check if our prediction matches the truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  #list of booleans

#Determine what fraction of the list of booleans is true..
#cast to floats and take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print the accuracy
print("Accuracy 1: %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))









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
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

###########################
# First Convolutional Layer
###########################
# First layer will consist of convolution, followed by max pooling
# convolutional will compute 32 features for each 5x5 patch 
	#Its weight tensor will have a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, 
	#the next is the number of input channels, and the last is the number of output channels. 
W_conv1 = weight_variable([5, 5, 1, 32])
# We will also have a bias vector with a component for each output channel
b_conv1 = bias_variable([32])

#To apply the layer, first reshape x to a 4d tensor, with the second and third dimensions corresponding 
#to image width and height, and the final dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1,28,28,1])

#convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


############################
# Second Convolutional Layer
############################
#To build a deep network, stack several layers of this type. 
#The second layer will have 64 features for each 5x5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#########################
# Densely Connected Layer
#########################
# image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#########
# Dropout
#########
#To reduce overfitting, we will apply dropout before the readout layer

#create a placeholder for the probability that a neuron's output is kept during dropout
keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

###############
# Readout Layer
###############
#Finally, we add a softmax layer, just like for the one layer softmax regression above.
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)





##############################
# Train and Evaluate the Model
##############################

#use code that is nearly identical to that for the simple one layer SoftMax network above
#differences are: 
	#replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer
	#include the additional parameter keep_prob in feed_dict to control the dropout rate
	#log every 100th iteration in the training process

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))




















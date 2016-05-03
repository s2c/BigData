from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#To use TensorFlow, we need to import it.
import tensorflow as tf

#create a symbol variable to use when describing the external operations
#x isn't a specific value, its a placeholder... 
#we'll input the value when we ask tensorflow to run a computation
#x will represent a 2-D tensor of floats... so we can input any number of MNIST images, each flattened into 784-dim vector
x = tf.placeholder(tf.float32, [None, 784]) #None means dim can be any length


#also need weights and biases for the model. Instead of using inputs though, use variables
#give tf.Variable the initial value of the variable... in this case tensors full of zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#implement the model... in just one line !
#first, multiply x by W with the expression tf.matmul(x, W)
#next, add b
#finally apply tf.nn.softmax
y = tf.nn.softmax(tf.matmul(x, W) + b)





############
#Train Model
############
#use very common, very nice cost function called "cross-entropy"

#placeholder to input correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

#implement cross entroy as the -Sum(y'log(y))
#note, tf.reduce_sum adds the elements in the 2nd dim of y, specified by reduction_indices=[1] parameter
#tf.reduce_mean computes the mean over all the examples in the batch
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Initialize the variables we created
init = tf.initialize_all_variables()


#launch the model in a Session, and run the operation that initializes the variables
sess = tf.Session()
sess.run(init)

#Let's train -- we'll run the training step 1000 times!
#Each step of the loop, we get a "batch" of one hundred random data points from our training set
#We run train_step feeding in the batches data to replace the placeholders.
#Using small batches of random data is called stochastic training (stochastic gradient descent here)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


###############
#Evaluate Model
###############

#tf.argmax gives you the index of the highest entry in a tensor along some axis
#tf.argmax(y,1) is the label our model thinks is most likely for each input
#tf.argmax(y_,1) is the correct label
#use tf.equal to check if our prediction matches the truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  #list of booleans

#Determine what fraction of the list of booleans is true..
#cast to floats and take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print the accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


































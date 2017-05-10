

################################################################################
#
#   Tensorflow - Simple Digit Recognition Model
#
################################################################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import Data
mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])         # Initialize image vector: A 784-dimensional vector (28 pixels x 28 pixels = 784)
W = tf.Variable(tf.zeros([784, 10]))                # Initialize Weights as zeros: 784 image vector multiplied by 10 classes (digits 0,1,2...9)
b = tf.Variable(tf.zeros([10]))                     # Initialize Digits as zeros:  10 digits (0,1,2,3,4,5,6,7,8,9)

y = tf.nn.softmax(tf.matmul(x, W) + b)              # Define softmax model equation:   y = softmax( x*W + b )


################################################################
#  Setup cross-entropy evaluator to determine model loss
################################################################
y_ = tf.placeholder(tf.float32, [None, 10])         # Initialize placeholder to accept the correct labels/targets

# Define cross entropy equation
# Cross-entropy gives us a way to express how different two probability distributions are
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))



################################################################
#  Train Model to minimize cross-entropy (minimize model loss)
################################################################
# Minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5
# Gradient descent enables TensorFlow to simply shifts each variable a little bit in the direction that reduces the cost.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# Create an operation to initialize the variables we created
init = tf.global_variables_initializer()


# Launch Session and run operation
sess = tf.Session()
sess.run(init)


# Train our model based on train_step 
# Run 1000 iterations
# Each step of the loop, we get a "batch" of one hundred random data points from our training set. 
# Rerun train_step feeding in the batches data to replace the placeholders.
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})



################################################################
# Evaluate Model Results
################################################################
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))          # Outputs a set of booleans (1=correct, 0=incorrect prediction)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      # Calculate accuracy % based on boolean correct_prediction array
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))




#ZEND
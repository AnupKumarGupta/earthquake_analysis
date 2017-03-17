import numpy as np
import pandas as pd
import tensorflow as tf

data_frame = pd.read_csv("earthquake_database.csv")
X_input = data_frame[['Latitude', 'Longitude']].as_matrix()
Y_input = data_frame['Magnitude'].as_matrix()

X_min = np.amin(X_input, 0)
X_max = np.amax(X_input, 0)
Y_min = np.amin(Y_input)
Y_max = np.amax(Y_input)

X_norm = (X_input - X_min) / (X_max - X_min)
Y_norm = Y_input

X_features = 2  # Number of input features
Y_features = 1  # Number of output features
samples = 23000  # Number of samples

# InputX1_reshape = np.resize(X_norm,(samples,X_features))
# InputY1_reshape = np.resize(Y_norm,(samples,Y_features))

# Training data
batch_size = 20000
X_train = X_norm[0:batch_size]
Y_train = Y_norm[0:batch_size]

# Validation data
validation_size = 2500
X_validate = X_norm[batch_size:batch_size + validation_size]
Y_validate = Y_norm[batch_size:batch_size + validation_size]

# Network hyper parametres
learning_rate = 0.001
training_iterations = 100000
display_iterations = 20000

# Input
X = tf.placeholder(tf.float32, shape=(None, X_features))  # [batch size, input_features]
# Output
Y = tf.placeholder(tf.float32)

# Neurons
layer_1 = 2  # Number of neurons in 1st layer
layer_2 = 2  # Number of neurons in 2nd layer
layer_3 = 2  # Number of neurons in 2nd layer

# Layer1 weights
weight_layer_1 = tf.Variable(tf.random_uniform([X_features, layer_1]))
# [input_features,Number of neurons])
bias_layer_1 = tf.Variable(tf.constant(0.1, shape=[layer_1]))

# Layer2 weights
weight_layer_2 = tf.Variable(tf.random_uniform([layer_1, layer_2]))
# [Number of neurons in preceding layer,Number of neurons])
bias_layer_2 = tf.Variable(tf.constant(0.1, shape=[layer_2]))

# Layer3 weights
weight_layer_3 = tf.Variable(tf.random_uniform([layer_2, layer_3]))
# [Number of neurons in preceding layer,Number of neurons])
bias_layer_3 = tf.Variable(tf.constant(0.1, shape=[layer_3]))

# Output layer weights
weight_layer_output = tf.Variable(tf.random_uniform([layer_3, Y_features]))
# [Number of neurons in preceding layer,output_features])
bias_layer_output = tf.Variable(tf.constant(0.1, shape=[Y_features]))

# Layer 1
hidden_layer_1 = tf.nn.relu(tf.matmul(X, weight_layer_1) + bias_layer_1)  # ReLU activation
# Layer 2
hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weight_layer_2) + bias_layer_2)  # ReLU activation
# Layer 3
hidden_layer_3 = tf.nn.relu(tf.matmul(hidden_layer_2, weight_layer_3) + bias_layer_3)  # ReLU activation
# Output layer
output_layer = tf.matmul(hidden_layer_3, weight_layer_output) + bias_layer_output  # linear activation

# Loss function
loss_mean_square = tf.reduce_mean(tf.square(Y - output_layer))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_mean_square)

# Operation to save variables
saver = tf.train.Saver()

# Initialization and session

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("Training loss:", sess.run([loss_mean_square], feed_dict={X: X_train, Y: Y_train}))

    for i in range(training_iterations):
        sess.run([train_step], feed_dict={X: X_train, Y: Y_train})
        if i % display_iterations == 0:
            print("Training loss is:", sess.run([loss_mean_square], feed_dict={X: X_train, Y: Y_train}),
                  "at iteration:", i)

            print("Validation loss is:", sess.run([loss_mean_square], feed_dict={X: X_validate, Y: Y_validate}),
                  "at iteration:", i)

    # Save the variables to disk.
    save_path = saver.save(sess, "/temp/earthquake_model.ckpt")
    print("Model saved in file: %s" % save_path)

    print("Final training loss:", sess.run([loss_mean_square], feed_dict={X: X_input, Y: Y_train}))
    print("Final validation loss:", sess.run([loss_mean_square], feed_dict={X: X_validate, Y: Y_validate}))


# Testing
latitude = input("Enter Latitude between -77 to 86:")
longitude = input("Enter Longitude between -180 to 180:")
depth = input("Enter Depth between 0 to 700:")
InputX2 = np.asarray([[latitude, longitude, depth]], dtype=np.float32)
InputX2_norm = (InputX2 - X_min) / (X_max - X_min)
InputX1test = np.resize(InputX2_norm, (1, X_features))

with tf.Session() as sess:
    # Restore variables from disk for validation.
    saver.restore(sess, "/temp/earthquake_model.ckpt")
    print("Model restored.")
    # print("Final validation loss:",sess.run([mean_square],feed_dict={X:InputX1v,Y:InputY1v}))
    print("output:", sess.run([output_layer], feed_dict={X: InputX1test}))

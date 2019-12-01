from __future__ import absolute_import, division, print_function, unicode_literals

"""
Charlie Wilkin, 11/28/19
EE456 Project

This script defines, trains, and saves a neural network for recognizing 
hand-written digits. The network consists of both convolutional and dense
layers, and reliably achieves greater than 99% accuracy.
"""

# Using tensorflow for its built-in hardware acceleration.
import tensorflow as tf

# Load the MNIST dataset of handwritten digits.
mnist = tf.keras.datasets.mnist

# Prepare a train-test split.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Adding a blank dimension because CNNs in Tensorflow take 4D inputs.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


# Define the structure of the model sequentially
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Adam is a modern, adaptive variant of traditional gradient descent.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to the data, making 5 passes over the dataset.
model.fit(x_train, y_train, epochs=1)

# Testing the accuracy of the model on data it's never seen before.
model.evaluate(x_test,  y_test, verbose=2)

# Save the model.
#model.save('trained_model')

model.summary()

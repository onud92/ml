# use tensorflow library
import tensorflow as tf
# Keras is TensorFlow's high-level API for deep learning
from tensorflow import keras
# use math library numpy
import numpy as np
# use graphing library Matplotlib
import matplotlib.pyplot as plt
# use Python's math library
import math

# generate this many sample datapoints
SAMPLES = 1000

# set seed value for generating same random numbers
SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

# generate a uniformly distributed set of random numbers in the range from
# 0 to 2π, which covers a complete sine wave oscillation
x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)

# Shuffle the values to guarantee they're not in order
np.random.shuffle(x_values)


# Calculate the corresponding sine values
y_values = np.sin(x_values)

# Add a small random number to each y value
y_values += 0.1 * np.random.randn(*y_values.shape)

# use 60% of data for training, 20% for testing and 20% for validation
TRAIN_SPLIT = int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

# chop data into three parts
x_train, x_validate, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_validate, y_test = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

# Double check that our splits add up correctly
assert (x_train.size + x_validate.size + x_test.size) == SAMPLES

# We'll use Keras to create a simple model architecture
model_1 = tf.keras.Sequential()

# First layer takes a scalar input and feeds it through 8 "neurons". The
# neurons decide whether to activate based on the 'relu' activation function.
model_1.add(keras.layers.Dense(16, activation='relu', input_shape=(1,)))
# Final layer is a single neuron, since we want to output a single value
model_1.add(keras.layers.Dense(1))
# Compile the model using the standard 'adam' optimizer and the mean squared error or 'mse' loss function for regression.
model_1.compile(optimizer='adam', loss='mse', metrics=['mae'])

model_1.summary()

# train the model now
history_1 = model_1.fit(x_train, y_train, epochs=500, batch_size=16, validation_data=(x_validate, y_validate))

# make predictions
predictions = model_1.predict(x_train)

# Plot the predictions along with the test data
plt.clf()
plt.title('Training data predicted vs actual values')
plt.plot(x_test, y_test, 'b.', label='Actual')
plt.plot(x_train, predictions, 'r.', label='Predicted')
plt.legend()
plt.savefig("pics/split_data_sinus.png") 
print("Grafik wurde als 'ergebnis_grafik.png' gespeichert!")

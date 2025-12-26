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
x_values = np.random.uniform(low=0, high=2 * math.pi, size=SAMPLES)

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
model_1.add(keras.layers.Dense(16, activation="relu", input_shape=(1,)))

model_1.add(keras.layers.Dense(16, activation="relu"))

# Final layer is a single neuron, since we want to output a single value
model_1.add(keras.layers.Dense(1))
# Compile the model using the standard 'adam' optimizer and the mean squared error or 'mse' loss function for regression.
model_1.compile(optimizer="adam", loss="mse", metrics=["mae"])

model_1.summary()

# train the model now
history_1 = model_1.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=(x_validate, y_validate),
)

# make predictions
predictions = model_1.predict(x_train)

# convert the model to the TensorFlow Lite formate without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model_1)
tflite_model = converter.convert()

# save model without quantization to the disk
with open("tflite_models/sine_model.tflite", "wb") as f:
    f.write(tflite_model)

# Convert the model to the TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model_1)

# Indicate that we want to perform the default optimizations,
# which include quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]


# Define a generator function that provides our test data's x values
# as a representative dataset, and tell the converter to use it
def representative_dataset_generator():
    for value in x_test:
        # Each scalar value must be inside of a 2D array that is wrapped in a list
        yield [np.array(value, dtype=np.float32, ndmin=2)]


converter.representative_dataset = representative_dataset_generator

# Convert the model
tflite_model = converter.convert()

# Save the model with quantization to disk
with open("tflite_models/sine_model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

# instantiate an interpreter for each model
sine_model = tf.lite.Interpreter("tflite_models/sine_model.tflite")
sine_model_quantized = tf.lite.Interpreter("tflite_models/sine_model_quantized.tflite")

# allocate memory for each model
sine_model.allocate_tensors()
sine_model_quantized.allocate_tensors()

# get indexes of the input and output tensors
sine_model_input_index = sine_model.get_input_details()[0]["index"]
sine_model_output_index = sine_model.get_output_details()[0]["index"]

sine_model_quantized_input_index = sine_model_quantized.get_input_details()[0]["index"]
sine_model_quantized_output_index = sine_model_quantized.get_output_details()[0][
    "index"
]

# Create arrays to store the results
sine_model_predictions = []
sine_model_quantized_predictions = []

# Run each model's interpreter for each value and store the results in arrays
for x_value in x_test:
    # Create a 2D tensor wrapping the current x value
    x_value_tensor = tf.convert_to_tensor([[x_value]], dtype=np.float32)
    # Write the value to the input tensor
    sine_model.set_tensor(sine_model_input_index, x_value_tensor)
    # Run inference
    sine_model.invoke()
    # Read the prediction from the output tensor
    sine_model_predictions.append(sine_model.get_tensor(sine_model_output_index)[0])

    # Do the same for the quantized model
    sine_model_quantized.set_tensor(sine_model_quantized_input_index, x_value_tensor)
    sine_model_quantized.invoke()
    sine_model_quantized_predictions.append(
        sine_model_quantized.get_tensor(sine_model_quantized_output_index)[0]
    )

# Plot the predictions along with the test data
plt.clf()
plt.title("Comparison of various models against actual values")
plt.plot(x_test, y_test, "bo", label="Actual")
plt.plot(x_train, predictions, "r.", label="Original Predictions")
plt.plot(x_test, sine_model_predictions, "bx", label="Lite predictions")
plt.plot(
    x_test, sine_model_quantized_predictions, "gx", label="Lite quantized predictions"
)
plt.legend()
plt.savefig("pics/pred_comparison.png")
print("Grafik wurde als 'pred_comparison.png' gespeichert!")

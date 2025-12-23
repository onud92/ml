# use tensorflow library
import tensorflow as tf
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


# Plot our data. The 'b.' argument tells the library to print blue dots.
plt.plot(x_values, y_values, 'b.')
plt.show()

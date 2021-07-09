import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Remove tf error warnings
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Solution for memory errors in TF

# Initialization of Tensors
x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
x = tf.constant([[1, 2, 3], [4, 5, 6]])
x = tf.ones((3, 3))
x = tf.zeros((2, 3))
x = tf.eye(3)
x = tf.random.normal((3, 3), mean=0, stddev=1)
x = tf.random.uniform((1, 3), minval=0, maxval=1)
x = tf.range(start=1, limit=10, delta=2)  # Same as python range function

# Changing datatype (Typecasting)
x = tf.cast(x, dtype=tf.float32)

# Mathematical operations
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])
z = tf.add(x, y) # Add operation
z = tf.subtract(x, y) # Subtract operation
z = tf.divide(x, y) # Element-wise division
z = tf.multiply(x, y) # Element-wise multiplication
z = tf.tensordot(x, y, axes=1)  # Dot product preferred way
z = tf.reduce_sum(x*y, axis=0)  # Dot product naive way
x = tf.random.normal((2, 3))
y = tf.random.normal((3, 4))
z = x @ y  # Matrix multiplication naive way
z = tf.matmul(x, y) # Matrix multiplication preferred way

# Indexing
x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
indices = tf.constant([0, 3])
x_ind = tf.gather(x, indices)   # Making a new tensor with multiple indices from x

# Reshaping
x = tf.range(9)
x = tf.reshape(x, (3, 3))
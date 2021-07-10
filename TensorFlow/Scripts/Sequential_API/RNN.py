import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Remove tf error warnings
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Solution for memory errors in TF

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0   # As float64 is unnecessary computation
x_test = x_test.astype("float32") / 255.0

model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(
    layers.SimpleRNN(512, return_sequences=True, activation='tanh')
)
model.add(layers.SimpleRNN(512, activation='relu'))
model.add(layers.Dense(10))

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=1e-3),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test)

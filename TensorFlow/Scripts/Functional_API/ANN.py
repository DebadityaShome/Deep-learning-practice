import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Remove tf error warnings
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)  # Solution for memory errors in TF

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flattening from (60000, 28, 28) to (60000, 784) and normalizing
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# Functional API (bit more flexible than sequential)
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu', name='second_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # From logits is set to True to make sure the output of the Dense layers is passed through a softmax first
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test)
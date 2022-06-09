from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define the neural network
def create_model(input_size, output_size, conv_size, dropout):
    model = Sequential()
    model.add(Conv2D(1, (1, conv_size), input_shape=(1, input_size, 1), activation="relu"))
    if dropout > 0:
        model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(output_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model

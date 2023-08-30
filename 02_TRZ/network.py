"""Convolutional Neural Network layer structure for TRZ"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, ZeroPadding2D, GRU, Bidirectional, Input, Reshape

def get_model(n_classes):
    # building a linear stack of layers with the sequential model
    model = Sequential()

    # First convolutional layer with max pooling
    model.add(Conv2D(8, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

    # Second convolutional layer with max pooling
    model.add(Conv2D(16, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

    # Hidden layer with 512 nodes
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))

    # Output layer with 32 nodes (one for each possible letter/number we predict)
    model.add(Dense(n_classes, activation="softmax"))

    # Ask Keras to build the TensorFlow model behind the scenes
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model



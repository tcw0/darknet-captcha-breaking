"""Convolutional Neural Network layer structure"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam

def get_model(n_classes):
    # building a linear stack of layers with the sequential model
    model = Sequential()

    # convolutional layer
    model.add(Conv2D(8, kernel_size=(5, 5), activation='relu'))
    # model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    # model.add(Conv2D(50, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
    # model.add(Conv2D(100, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    # model.add(Conv2D(200, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

    # flatten output of conv
    model.add(Flatten())

    # hidden layer
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.2))

    # output layer
    model.add(Dense(n_classes, activation='softmax'))

    # compiling the sequential model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model



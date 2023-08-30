"""Train network and store trained model for chosen subfolder"""
import os
import numpy as np
import cv2
import pickle
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import network
import config
import TRZ.prepare_training
import WHM.prepare_training

from tensorflow.keras import callbacks


def train_model(subfolder):
    USE_GPU = False     # Change to True when using GPU-KONG
    if USE_GPU:
        GPU_COUNT = 2
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_COUNT)

    # Get training and validation data from specified subfolder
    if subfolder == 'WHM':
        X_train, X_test, Y_train, Y_test = WHM.prepare_training.training_val_data()
    elif subfolder == 'TRZ':
        X_train, X_test, Y_train, Y_test = TRZ.prepare_training.training_val_data()
    else:
        raise ValueError("Please provide the name to a valid subfolder!")

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_test = lb.transform(Y_test)

    # Save the mapping from labels to one-hot encodings.
    with open(os.path.join(os.path.dirname(__file__), subfolder, config.MODEL_LABELS_FILENAME), "wb") as f:
        pickle.dump(lb, f)

    model = network.get_model(len(lb.classes_))

    earlystopping = callbacks.EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=5)
    modelcheckpoint = callbacks.ModelCheckpoint(os.path.join(os.path.dirname(__file__), subfolder, f"models{os.sep}model100"), monitor="val_accuracy", mode="max", verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, callbacks=[earlystopping, modelcheckpoint])

    print("Finished fitting")


if __name__ == '__main__':
    train_model(subfolder="TRZ")


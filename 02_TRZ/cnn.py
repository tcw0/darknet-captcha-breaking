"""Train network and store trained model for TRZ"""
import os
import numpy as np
import cv2
import pickle
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import network
import config
import helpers

from tensorflow.keras import callbacks


def train_model():
    USE_GPU = False     # Change to True when using GPU-KONG
    if USE_GPU:
        GPU_COUNT = 2
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_COUNT)


    data = []
    labels = []

    # Iterate over training images
    for image_file in paths.list_images(os.path.join(os.path.dirname(__file__), f"data{os.sep}training_data")):
        # Load image and convert to grayscale
        image = cv2.imread(image_file)
        cv2.imwrite(f"test1.png", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"test2.png", image)

        # Resize letter to 35x35 (wxh) pixels
        image = helpers.resize_to_fit(image, 35, 35)
        cv2.imwrite(f"test3.png", image)

        # Add third channel dimension for Keras
        image = np.expand_dims(image, axis=2)

        # Label letter based on folder
        label = image_file.split(os.path.sep)[-2]

        # Add letter image and label to training data
        data.append(image)
        labels.append(label)


    # Divide by 255 to normalize to range 0 to 1 (pixel values range from 0 to 256)
    data = np.array(data, dtype=np.float32) / 255.0
    labels = np.array(labels)

    # Split into training and validation data
    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.2, random_state=0)

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_test = lb.transform(Y_test)

    # Save the mapping from labels to one-hot encodings.
    with open(config.MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    model = network.get_model(len(lb.classes_))

    earlystopping = callbacks.EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=5)
    modelcheckpoint = callbacks.ModelCheckpoint(os.path.join(os.path.dirname(__file__), f"models{os.sep}model6"), monitor="val_accuracy", mode="max", verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=25, batch_size=32, callbacks=[earlystopping, modelcheckpoint])

    print("Finished fitting")


if __name__ == '__main__':
    train_model()


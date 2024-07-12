import pandas as pd
import numpy as np

from keras.preprocessing import image

from keras import regularizers

from keras.optimizers import RMSprop, Adam, Optimizer, Optimizer
# MODEL LAYERS
import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D, \
    BatchNormalization, \
    Permute, TimeDistributed, Bidirectional, GRU, SimpleRNN, LSTM, GlobalAveragePooling2D, SeparableConv2D
from keras import models
from keras import layers
import tensorflow as tf
from keras.applications import VGG16, VGG19, inception_v3
from keras import backend as K
from keras.utils import plot_model
# SKLEARN CLASSIFIER


# IGNORING WARNINGS
from warnings import filterwarnings

filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
import keras.utils as image
import numpy as np
import keras.utils as image


def working(user_file):
    print(user_file)
    cnn = tf.keras.models.load_model("/Users/jaiharishsatheshkumar/PycharmProjects/newmini/website-main/backend"
                                     "/parkinson_modelconfig/parkinson.h5")
    prediction = ""

    test_image = image.load_img(user_file, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)  # what does this do?
    result = cnn.predict(test_image / 255)
    if result[0][0] > 0.5:
        prediction = 'healthy'
    else:
        prediction = 'parkinson'
    return prediction


class parkinsontest:
    def __init__(self):
        self.cnn = tf.keras.models.Sequential()
        self.training_set = ""
        self.test_set = ""

    def processdata(self):
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                        shear_range=0.2,  # tilt of the images upto 20%
                                                                        zoom_range=0.2,
                                                                        horizontal_flip=True)
        self.training_set = train_datagen.flow_from_directory(
            "/Users/jaiharishsatheshkumar/PycharmProjects/newmini/website-main/backend/spiral/training",
            target_size=(64, 64),  # changing all the images to this size
            batch_size=32,
            # in each batch there will be a total of 32 images
            class_mode='binary')  # one or zero

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255)  # we are dividing by 255 to bring it into the range of 0 to 1
        self.test_set = test_datagen.flow_from_directory(
            "/Users/jaiharishsatheshkumar/PycharmProjects/newmini/website-main/backend/spiral/testing",
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

    def cnnlayers(self):
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))  # 2X2 pooling, stride- 2 boxes at a time

        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        self.cnn.add(tf.keras.layers.Flatten())

        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))  # dense - adding neurons (128 neurons)

        self.cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        self.cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def building(self):
        self.processdata()
        self.cnnlayers()
        self.cnn.fit(x=self.training_set, validation_data=self.test_set,
                     epochs=25)  # epochs- num of times, it has to run
        self.cnn.save(
            "/Users/jaiharishsatheshkumar/PycharmProjects/newmini/website-main/backend/parkinson_modelconfig/parkinson.h5")



"""
train_cnn.py

Trains CNNs

Copyright (C) 2018, Akhilan Boopathy <akhilan@mit.edu>
                    Lily Weng  <twweng@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Luca Daniel <dluca@mit.edu>
"""
import csv
import time

from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, BatchNormalization
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.optimizers import Adam

import tensorflow as tf
import os


def train(data, file_name, filters, kernels, num_epochs=50, batch_size=128, train_temp=1, init=None, activation=tf.nn.relu, bn=False, use_padding_same=False,
          use_early_stopping=True):
    """
    Train a n-layer CNN for MNIST and CIFAR
    """
    # create a Keras sequential model
    model = Sequential()
    if use_padding_same:
        model.add(Conv2D(filters[0], kernels[0], input_shape=data.train_data.shape[1:], padding="same"))
    else:
        model.add(Conv2D(filters[0], kernels[0], input_shape=data.train_data.shape[1:]))
    if bn:
        model.add(BatchNormalization())
    model.add(Activation(activation))
    # model.add(Lambda(activation))
    for f, k in zip(filters[1:], kernels[1:]):
        if use_padding_same:
            model.add(Conv2D(f, k, padding="same"))
        else:
            model.add(Conv2D(f, k))
        if bn:
            model.add(BatchNormalization())
        model.add(Activation(activation))
        # ReLU activation
        # model.add(Lambda(activation))
    # the output layer, with 10 classes
    model.add(Flatten())
    model.add(Dense(data.train_labels.shape[1]))

    # load initial weights when given
    if init != None:
        model.load_weights(init)

    # define the loss function which is the cross entropy between prediction and true label
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted / train_temp)

    # initiate the Adam optimizer
    sgd = Adam()

    # compile the Keras model, given the specified loss and optimizer
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.summary()

    if data.dataset == "tiny-imagenet-200" or "GTSRB":
        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode="nearest")
    else:
        datagen = ImageDataGenerator()

    datagen.fit(data.train_data)

    print("Traing a {} layer model, saving to {}".format(len(filters) + 1, file_name), flush=True)

    start_time = time.time()
    if use_early_stopping:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True,
                                                          verbose=1)
        history = model.fit_generator(datagen.flow(data.train_data, data.train_labels, batch_size=batch_size),
                            validation_data=(data.validation_data, data.validation_labels),
                            epochs=400,
                            shuffle=True,
                            callbacks=[early_stopping],
                            verbose=0)
        best_epoc = len(history.history['loss']) - early_stopping.wait

    else:
        history = model.fit(data.train_data, data.train_labels,
                            batch_size=batch_size,
                            validation_data=(data.validation_data, data.validation_labels),
                            epochs=num_epochs,
                            shuffle=True,
                            verbose=0)
        best_epoc = len(history.history['loss'])
    time_taken = (time.time() - start_time)

    # run training with given dataset, and print progress

    num_ephocs_trained = len(history.history['loss'])
    metafile = "output/models_meta.csv"
    if not os.path.exists(metafile):
        with open(metafile, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["num_epochs", "best_epoch", "time_taken", "time_per_epoch", "accuracy", "file_name"])
    with open(metafile, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [num_ephocs_trained, best_epoc, time_taken, float(time_taken) / float(num_ephocs_trained), history.history["val_acc"][best_epoc], file_name])

    print("saving - ", file_name)
    # save model to a file
    if file_name != None:
        is_saved = False
        while not is_saved:
            try:
                model.save(file_name)
                is_saved = True
            except Exception as e:
                print("could not save model: ", e)
                time.sleep(5)

    return history

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
import gc
import threading
import time
import tensorflow as tf

from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, BatchNormalization
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.optimizers import Adam, SGD

import os

from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_dynamic_keras_config(tf):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    return config

def train(data, file_name, filters, kernels, num_epochs=50, batch_size=128, train_temp=1, init=None, activation=tf.nn.relu, bn=False, use_padding_same=False,
          use_early_stopping=True):
    """
    Train a n-layer CNN for MNIST and CIFAR
    """
    # create a Keras sequential model
    sess = tf.Session(config=get_dynamic_keras_config(tf))
    with sess.as_default():
        model = create_model(activation, bn, data, filters, init, kernels, use_padding_same)

        # define the loss function which is the cross entropy between prediction and true label
        def fn(correct, predicted):
            return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                           logits=predicted / train_temp)

        patience = 30
        optimizer = Adam()
        monitor = 'val_loss'
        min_delta = 0
        if data.dataset == "cifar100":
            patience = 50
        elif data.dataset == "GTSRB":
            optimizer = Adam(lr=0.0005)
        elif data.dataset == "caltech_siluettes":
            patience = 50
        elif data.dataset =="dogs-and-cats":
            monitor = "loss"


        # compile the Keras model, given the specified loss and optimizer

        devices = get_available_gpus()

        if len(devices) >= 2:
            print(f"using {len(devices)} gpus")
            model = tf.keras.utils.multi_gpu_model(model, gpus=len(devices))

        model.compile(loss=fn,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.summary()

        datagen = get_data_augmenter(data)

        print("Traing a {} layer model, saving to {}".format(len(filters) + 1, file_name), flush=True)

        start_time = time.time()
        if use_early_stopping:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True,
                                                              verbose=1, min_delta=min_delta)
            flow = datagen.flow(data.train_data, data.train_labels, batch_size=batch_size)
            history = model.fit_generator(flow,
                                          validation_data=(data.validation_data, data.validation_labels),
                                          epochs=400,
                                          shuffle=True,
                                          callbacks=[early_stopping],
                                          verbose=1,
                                          max_queue_size=batch_size,
                                          workers=12,
                                          use_multiprocessing=True)
            best_epoc = len(history.history['loss']) - early_stopping.wait

        else:
            history = model.fit(data.train_data, data.train_labels,
                                batch_size=batch_size,
                                validation_data=(data.validation_data, data.validation_labels),
                                epochs=num_epochs,
                                shuffle=True,
                                verbose=1)
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
                [num_ephocs_trained, best_epoc, round(time_taken, 1), round(float(time_taken) / float(num_ephocs_trained), 1),
                 history.history["val_acc"][best_epoc - 1], file_name])

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
    sess.close()
    gc.collect()
    return history


def create_model(activation, bn, data, filters, init, kernels, use_padding_same):
    model = Sequential()
    if use_padding_same:
        model.add(Conv2D(filters[0], kernels[0], input_shape=data.train_data.shape[1:], padding="same"))
    else:
        model.add(Conv2D(filters[0], kernels[0], input_shape=data.train_data.shape[1:]))
    if bn:
        apply_bn(data, model)
    # model.add(Activation(activation))
    model.add(Lambda(activation))
    for f, k in zip(filters[1:], kernels[1:]):
        if use_padding_same:
            model.add(Conv2D(f, k, padding="same"))
        else:
            model.add(Conv2D(f, k))
        if bn:
            apply_bn(data, model)
        # model.add(Activation(activation))
        # ReLU activation
        model.add(Lambda(activation))
    # the output layer, with 10 classes
    model.add(Flatten())
    model.add(Dense(data.train_labels.shape[1]))
    # load initial weights when given
    if init != None:
        model.load_weights(init)
    return model


def get_data_augmenter(data):
    if data.dataset == "GTSRB":
        print("datagen1")
        datagen = ImageDataGenerator(
            rotation_range=5,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode="nearest")
    elif data.dataset == "caltech_siluettes":
        print("datagen2")
        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode="nearest")
    elif data.dataset == "tiny-imagenet-200" \
            or data.dataset =="cifar100" \
            or data.dataset =="cifar" \
            or data.dataset =="dogs-and-cats":
        print("datagen3")
        datagen = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode="nearest")
    elif data.dataset == "sign-language":
        print("datagen4")
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            vertical_flip=False,
            fill_mode="nearest")
    else:
        print("no augmentation")
        datagen = ImageDataGenerator()
    return datagen


def apply_bn(data, model):
    if data.dataset == "cifar" or "caltech_siluettes" or "cifar100" or "tiny-imagenet-200":
        model.add(BatchNormalization(momentum=0.9))
    elif data.dataset == "GTSRB":
        model.add(BatchNormalization(momentum=0.8))
    else:
        model.add(BatchNormalization())

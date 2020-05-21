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
from tensorflow.contrib.keras.api.keras.utils import Sequence
from tensorflow.contrib.keras.api.keras.backend import set_session
from tensorflow.contrib.keras.api.keras.backend import get_session
import numpy as np

import os

from tensorflow.python.client import device_lib

from Attacks.PGD_attack import LinfPGDAttack


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_dynamic_keras_config(tf):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    return config


def train(data, file_name, filters, kernels, num_epochs=50, batch_size=128, train_temp=1, init=None, activation=tf.nn.relu, bn=False, use_padding_same=False,
          use_early_stopping=True, train_on_adversaries=False):
    """
    Train a n-layer CNN for MNIST and CIFAR
    """
    # create a Keras sequential model
    sess = tf.Session(config=get_dynamic_keras_config(tf))
    with sess.as_default():
        with tf.get_default_graph().as_default():
            set_session(sess)
            model = create_model(activation, bn, data, filters, init, kernels, use_padding_same)

            # define the loss function which is the cross entropy between prediction and true label
            def fn(correct, predicted):
                return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                               logits=predicted / train_temp)

            min_delta, optimizer, patience = get_training_parameters(data)

            model.compile(loss=fn,
                          optimizer=optimizer,
                          metrics=['accuracy'])


            model.summary()

            print("Traing a {} layer model, saving to {}".format(len(filters) + 1, file_name), flush=True)

            if train_on_adversaries:
                best_epoc, history, time_taken = train_adversarially(batch_size, data, min_delta, model, patience, use_early_stopping, sess)
                metafile = "output/adversarial_models_meta.csv"
            else:
                best_epoc, history, time_taken = train_normally(batch_size, data, min_delta, model, num_epochs, patience, use_early_stopping)
                metafile = "output/models_meta.csv"

            # run training with given dataset, and print progress

            num_ephocs_trained = len(history.history['loss'])

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


def train_normally(batch_size, data, min_delta, model, num_epochs, patience, use_early_stopping):
    datagen = get_data_augmenter(data)
    datagen.fit(data.train_data)
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
    return best_epoc, history, time_taken

def train_adversarially(batch_size, data, min_delta, model, patience, use_early_stopping, sess):
    train_datagen = AdversarialImagesSequence(data.train_data, data.train_labels, batch_size, model, data.dataset, sess)
    val_datagen = AdversarialImagesSequence(data.validation_data, data.validation_labels, batch_size, model, data.dataset, sess)
    start_time = time.time()
    if use_early_stopping:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True,
                                                          verbose=1, min_delta=min_delta)
        history = model.fit_generator(generator=train_datagen,
                                      validation_data=val_datagen,
                                      epochs=500,
                                      shuffle=True,
                                      callbacks=[early_stopping],
                                      verbose=1,
                                      max_queue_size=batch_size)
        best_epoc = len(history.history['loss']) - early_stopping.wait

    else:
        raise ValueError("not implemented")
    time_taken = (time.time() - start_time)
    return best_epoc, history, time_taken

class PDGModel():
    def __init__(self, model, x_set, y_set):
        image_size = x_set.shape[1]
        num_channels = x_set.shape[3]
        num_labels = y_set.shape[1]

        shape = (None, image_size, image_size, num_channels)
        x_input = tf.placeholder(tf.float32, shape)
        y_input = tf.placeholder(tf.float32, [None, num_labels])
        pre_softmax = model(x_input)

        y_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=pre_softmax)
        xent = tf.reduce_sum(y_loss)

        self.xent = xent
        self.x_input = x_input
        self.y_input = y_input


class AdversarialImagesSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, model, dataset_name, sess):

        pdg_model = PDGModel(model, x_set, y_set)
        epsilon = self.get_epsilon(dataset_name)
        self.dataset_name = dataset_name
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.model = model
        adv_steps = 20
        self.attack = LinfPGDAttack(pdg_model, epsilon, adv_steps, epsilon * 1.33 / adv_steps, random_start=True)
        self.sess = sess
        tf.keras.backend.get_session()

        self.datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest").flow(x_set, y_set, batch_size=batch_size)

    def get_epsilon(self, dataset):
        if dataset == "mnist":
            epsilon = 0.1
        elif dataset == "sign-language":
            epsilon = 0.03
        elif dataset == "caltech_siluettes":
            epsilon = 0.05
        elif dataset == "rockpaperscissors":
            epsilon = 0.04
        elif dataset == "cifar":
            epsilon = 0.01
        elif dataset == "GTSRB":
            epsilon = 0.05
        else:
            raise ValueError("Unkown dataset")
        return epsilon

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        if self.dataset_name == "caltech_siluettes" or self.dataset_name == "cifar":
            batch_x, batch_y = self.datagen.__getitem__(idx)
        else:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return self.attack.perturb(batch_x, batch_y, self.sess, verbose=False), np.array(batch_y)

def get_training_parameters(data):
    patience = 30
    optimizer = Adam()
    min_delta = 0.0001
    if data.dataset == "cifar100":
        patience = 50
    elif data.dataset == "GTSRB":
        optimizer = Adam(lr=0.0005)
        #  Senkes?
        # optimizer = Adam(lr=0.0003)
        patience = 50
    elif data.dataset == "caltech_siluettes":
        patience = 50
    elif data.dataset == "rockpaperscissors":
        patience = 50
    return min_delta, optimizer, patience


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
    # the output layer
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
        print("datagen3")
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
    elif data.dataset == "rockpaperscissors":
        print("datagen3")
        datagen = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest")
    elif data.dataset == "tiny-imagenet-200" \
            or data.dataset == "cifar100" \
            or data.dataset == "cifar" \
            or data.dataset == "dogs-and-cats":
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
    elif data.dataset == "sign-language"\
            or data.dataset == "mnist":
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
    if data.dataset == "cifar" \
            or data.dataset == "caltech_siluettes" \
            or data.dataset == "cifar100" \
            or data.dataset == "tiny-imagenet-200":
        model.add(BatchNormalization(momentum=0.9))
    elif data.dataset == "GTSRB" \
            or data.dataset == "caltech_siluettes":
        model.add(BatchNormalization(momentum=0.8))
    else:
        model.add(BatchNormalization())

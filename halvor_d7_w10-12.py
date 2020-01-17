import time

from train_cnn import train as train_cnn
from pymain import run_cnn
from setup_mnist import MNIST
import csv
import os.path
from Attacks.cw_attack import cw_attack
import time as timer
from tensorflow.contrib.keras.api.keras.models import load_model
import tensorflow as tf
from setup_mnist import MNIST
from enum import Enum
import logging

class NnArchitecture(Enum):
    ONLY_CNN = "only_cnn"


def get_name(dataset="mnist", nn_architecture="only_cnn", pooling="None", depth=1, width=8, filters=1, kernels=1, epochs=10,
             activation_function="relu",
             stride=1, bias=True, initializer="glorot_uniform", regulizer="None", batch_normalization=False,
             temperature=1,
             batch_size=128):
    return "models/dataset={}_nn_architecture={}_pooling={}_detph={}_width={}_filter={}_kernel={}_epochs={}_activationFunction={}_" \
           "stride={}_bias={}_initializer={}_regualizer={}_batchNormalization={}_temperature={}_batchSize={}" \
        .format(dataset, nn_architecture, pooling, depth, width, filters, kernels, epochs, activation_function, stride, bias,
                initializer, regulizer, batch_normalization, temperature, batch_size)


def make_result_file(name):
    if not os.path.exists(name):
        print("made new resultfile")
        with open(name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["dataset", "nn_architecture", "pooling", "depth", "width", "filter", "kernel", "epochs",
                             "activation_function", "stride", "has_bias",
                             "initializer", "regulizer", "has_batch_normalization", "temperature", "bach_size",
                             "lower_bound", "upper_bound", "l_norm", "time_elapsed", "accuracy", "file_name"])

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

def get_accuracy_of(file, batch_size):
    keras_model = load_model(file, custom_objects={'fn': fn, 'tf': tf})
    data = MNIST()
    loss, acc = keras_model.evaluate(data.validation_data, data.validation_labels, batch_size=batch_size)
    print("the accuracy is ", acc)
    return acc


def train_and_save_network(file_name, filters, kernels, epochs):
    sess = tf.keras.backend.get_session()
    tf.keras.backend.set_session(sess)
    history = train_cnn(MNIST(), file_name=file_name, filters=filters, kernels=kernels, num_epochs=epochs)
    accuracy = history.history["val_acc"][-1]
    tf.keras.backend.clear_session()
    return accuracy


def get_lower_bound(file_name, num_image, l_norm, only_cnn, activation_function):
    print(file_name)
    sess = tf.keras.backend.get_session()
    tf.keras.backend.set_session(sess)
    avg_lower_bound, total_time = run_cnn(file_name, num_image, l_norm, only_cnn, activation_function)
    tf.keras.backend.clear_session()
    return avg_lower_bound

def get_upper_bound(file_name, l_norm, num_image):
    sess = tf.keras.backend.get_session()
    tf.keras.backend.set_session(sess)
    upper_bound, time_taken = cw_attack(file_name, l_norm, sess, num_image=num_image)
    tf.keras.backend.clear_session()
    return upper_bound

def csv_contains_file(csv_file, file_name):
    with open(csv_file, "rt") as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')

        first_row = next(csvreader)
        column_number = 0
        for i in range(len(first_row)):
            if first_row[i] == "file_name":
                column_number = i
                break

        for row in csvreader:
            if row[column_number] == file_name:
                return True
    return False

def file_exists(file):
    return os.path.exists(file)

def main():
    dataset = "mnist"
    nn_architecture = NnArchitecture.ONLY_CNN.value
    pooling = "None"
    epochs = 10
    activation_function = "ada"
    stride = 1
    bias = True
    initializer = "glorot_uniform"
    regulizer = "None"
    batch_normalization = False
    temperature = 1
    batch_size = 128
    num_image = 10
    l_norm = "i"
    result_file = 'results/results_10_d7_sapmles.csv'
    make_result_file(result_file)
    logging.basicConfig(filename='log.log', level="ERROR")
    for activation_function in ["ada", "sigmoid", "tanh", "arctan"]:
        if activation_function == "sigmoid":
            tf_activation = tf.math.sigmoid
        if activation_function == "ada":
            tf_activation = tf.nn.relu
        if activation_function == "tanh":
            tf_activation = tf.math.tanh
        if activation_function == "arctan":
            tf_activation = tf.math.atan
        for depth in range(7, 8):
            for width_x in range(1, 13):
                width=int(2**width_x)
                for kernel_size in range(1, 15, 2):
                    for filter_size in range(3, 15, 2):

                        filters = [filter_size for i in range(depth)]
                        kernels = [kernel_size for i in range(depth)]

                        file_name = get_name(
                            dataset=dataset,
                            nn_architecture=nn_architecture,
                            pooling=pooling,
                            depth=depth,
                            width=width,
                            filters=filter_size,
                            kernels=kernel_size,
                            epochs=epochs,
                            activation_function=activation_function,
                            stride=stride,
                            bias=bias,
                            initializer=initializer,
                            regulizer=regulizer,
                            batch_normalization=batch_normalization,
                            temperature=temperature,
                            batch_size=batch_size)

                        start_time = timer.time()

                        have_results = False
                        if file_exists(file_name):
                            if not csv_contains_file(result_file, file_name):
                                accuracy = get_accuracy_of(file_name, batch_size)
                        else:
                            try:
                                accuracy = train_and_save_network(file_name, filters, kernels, epochs)
                            except Exception as e:
                                logging.exception("This file had an error: \n" + file_name + "\n" + str(e) + "\n\n")
                                continue


                        if csv_contains_file(result_file, file_name):
                            print("================================================")
                            print("skiped", file_name, "as the bounds was aready calculated")
                            print()
                            print("================================================")
                            continue
                        else:
                            if accuracy < 0.95:
                                print("================================================")
                                print("skiped", file_name, "as the accuracy was too low")
                                print()
                                print("================================================")
                                continue
                            else:
                                lower_bound = get_lower_bound(file_name, num_image, l_norm, nn_architecture==NnArchitecture.ONLY_CNN, activation_function)
                                try:
                                    upper_bound = None#get_upper_bound(file_name, l_norm, num_image)
                                except:
                                    upper_bound = None


                            time_elapsed = timer.time() - start_time
                            print("time elapsed", time_elapsed)

                            with open(result_file, 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                                writer.writerow([dataset, nn_architecture, pooling, depth, width, filter_size, kernel_size, epochs,
                                                activation_function, stride, bias, initializer, regulizer, batch_normalization,
                                                temperature, batch_size, lower_bound, upper_bound, l_norm, time_elapsed, accuracy
                                                 , file_name])




# hvilke paremetere kan jeg tweake på??
# cnn, ikke cnn, blanding
# pooling
# dybde nn
# bredde nn
# activation function
# stride
# filters
# kernels
# epochs
# dilation
# bias
# Initializers
# Regulizers
# Batch_normalization
# Temperature
# Batch size
main()
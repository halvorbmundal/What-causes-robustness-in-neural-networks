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

class NnArchitecture(Enum):
    ONLY_CNN = "only_cnn"


def get_name(dataset="mnist", nn_architecture="only_cnn", pooling="None", depth=1, width=8, filters=[0], kernels=[0], epochs=10,
             activation_function="relu",
             stride=1, bias=True, initializer="glorot_uniform", regulizer="None", batch_normalization=False,
             temperature=1,
             batch_size=128):
    return "models/dataset={}_nn_architecture={}_pooling={}_detph={}_width={}_filter={}_kernel={}_epochs={}_activationFunction={}_" \
           "stride={}_bias={}_initializer={}_regualizer={}_batchNormalization={}_temperature={}_batchSize={}" \
        .format(dataset, nn_architecture, pooling, depth, width, filters, kernels, epochs, activation_function, stride, bias,
                initializer, regulizer, batch_normalization, temperature, batch_size)


def make_result_file():
    if not os.path.exists('results/results.csv'):
        print("made new resultfile")
        with open('results/results.csv', 'a', newline='') as csvfile:
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
    print(acc)


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
    lower_bound, total_time = run_cnn(file_name, num_image, l_norm, only_cnn, activation_function)
    tf.keras.backend.clear_session()
    return lower_bound

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
    make_result_file()
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
    num_image = 1
    l_norm = "i"
    result_file = 'results/results.csv'
    for depth in range(1, 5):
        for width in range(8, 30, 10):
            for kernel_size in range(1, 5):
                for filter_size in range(1, 5):

                    filters = [filter_size for i in range(depth)]
                    kernels = [kernel_size for i in range(depth)]

                    file_name = get_name(
                        dataset=dataset,
                        nn_architecture=nn_architecture,
                        pooling=pooling,
                        depth=depth,
                        width=width,
                        filters=filters,
                        kernels=kernels,
                        epochs=1,
                        activation_function=activation_function,
                        stride=1,
                        bias=bias,
                        initializer=initializer,
                        regulizer=regulizer,
                        batch_normalization=batch_normalization,
                        temperature=temperature,
                        batch_size=batch_size)

                    start_time = timer.time()

                    if file_exists(file_name):
                        accuracy = get_accuracy_of(file_name, batch_size)
                    else:
                        accuracy = train_and_save_network(file_name, filters, kernels, epochs)

                    if csv_contains_file(result_file, file_name):
                        print("================================================")
                        print("skiped", file_name, "as the bounds was aready calculated")
                        print()
                        print("================================================")
                        continue
                    else:
                        lower_bound = get_lower_bound(file_name, num_image, l_norm, nn_architecture==NnArchitecture.ONLY_CNN, activation_function)
                        upper_bound = get_upper_bound(file_name, l_norm, num_image)

                        time_elapsed = timer.time() - start_time

                        with open(result_file, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow([dataset, nn_architecture, pooling, depth, width, filters, kernels, epochs,
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
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
import multiprocessing
import gc


class NnArchitecture(Enum):
    ONLY_CNN = "only_cnn"


def get_name(parameter_class):
    directory = "models2"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory + "/dataset={}_nn_architecture={}_pooling={}_detph={}_width={}_filter={}_kernel={}_epochs={}_activationFunction={}_" \
                       "stride={}_bias={}_initializer={}_regualizer={}_batchNormalization={}_temperature={}_batchSize={}" \
        .format(parameter_class.dataset, parameter_class.nn_architecture, parameter_class.pooling,
                parameter_class.depth,
                parameter_class.width, parameter_class.filter_size, parameter_class.kernel_size, parameter_class.epochs,
                parameter_class.activation_function_string, parameter_class.stride, parameter_class.bias,
                parameter_class.initializer, parameter_class.regulizer, parameter_class.has_batch_normalization,
                parameter_class.temperature, parameter_class.batch_size)


def make_result_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory + file):
        print("made new resultfile")
        with open(directory + file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["dataset", "nn_architecture", "pooling", "depth", "width", "filter", "kernel", "epochs",
                             "activation_function", "stride", "has_bias",
                             "initializer", "regulizer", "has_batch_normalization", "temperature", "bach_size",
                             "lower_bound", "upper_bound", "l_norm", "time_elapsed", "accuracy", "file_name"])


def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)


def setDynamicGPUAllocation():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    #config.log_device_placement = True  # to log device placement (on which device the operation ran)

    sess = tf.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

def get_accuracy_of(file, batch_size):
    keras_model = load_model(file, custom_objects={'fn': fn, 'tf': tf, 'atan': tf.math.atan})
    #TODO
    data = MNIST()
    loss, acc = keras_model.evaluate(data.validation_data, data.validation_labels, batch_size=batch_size)
    print("the accuracy is ", acc)
    return acc


def train_and_save_network(file_name, filters, kernels, epochs, tf_activation, batch_normalization):
    sess = tf.keras.backend.get_session()
    tf.keras.backend.set_session(sess)
    history = train_cnn(MNIST(),
                        file_name=file_name,
                        filters=filters,
                        kernels=kernels,
                        num_epochs=epochs,
                        activation=tf_activation,
                        bn=batch_normalization)
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


def get_tf_activation_function_from_string(activation_function_string):
    if activation_function_string == "sigmoid":
        return tf.math.sigmoid
    if activation_function_string == "ada":
        return tf.nn.relu
    if activation_function_string == "tanh":
        return tf.math.tanh
    if activation_function_string == "arctan":
        return tf.math.atan


def train_and_get_accuracy_of_nn(file_name, filters, kernels, tf_activation, has_batch_normalization):
    if file_exists(file_name):
        if csv_contains_file(CnnTestParameters.result_folder + CnnTestParameters.result_file, file_name):
            print("================================================")
            print("skiped", file_name, "as the bounds was aready calculated")
            print()
            print("================================================")
            return True, None
        else:
            keras_lock.acquire()
            try:
                return False, get_accuracy_of(file_name, CnnTestParameters.batch_size)
            except Exception as e:
                print("An exeption occured")
                logging.exception("This file had an error: \n" + file_name + "\n" + str(e) + "\n\n")
                return True, None
            finally:
                keras_lock.release()
    else:
        keras_lock.acquire()
        print("k-lock aquired")
        try:
            accuracy = train_and_save_network(file_name,
                                                 filters,
                                                 kernels,
                                                 CnnTestParameters.epochs,
                                                 tf_activation,
                                                 has_batch_normalization)
            print("accuracy: ", accuracy)
            return False, accuracy
        except Exception as e:
            print("An exeption occured")
            logging.exception("This file had an error: \n" + file_name + "\n" + str(e) + "\n\n")
            return True, None
        finally:
            print("k-lock released")
            keras_lock.release()


def calculate_lower_bound(accuracy, file_name, num_image, l_norm, nn_architecture, activation_function_string):
    if accuracy < 0.95:
        print("================================================")
        print("skiped", file_name, "as the accuracy was too low")
        print()
        print("================================================")
        return True, None
    else:
        return False, get_lower_bound(file_name,
                                      num_image,
                                      l_norm,
                                      nn_architecture == NnArchitecture.ONLY_CNN.value,
                                      activation_function_string)


"""try:
    upper_bound = None  # get_upper_bound(file_name, l_norm, num_image)
except:
    upper_bound = None"""


def write_to_file(parameters, lower_bound, accuracy, time_elapsed):
    write_lock.acquire()  # from global variable
    try:
        with open(parameters.result_folder + parameters.result_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                [parameters.dataset, parameters.nn_architecture, parameters.pooling, parameters.depth, parameters.width,
                 parameters.filter_size, parameters.kernel_size, parameters.epochs,
                 parameters.activation_function_string, parameters.stride, parameters.bias, parameters.initializer,
                 parameters.regulizer, parameters.has_batch_normalization,
                 parameters.temperature, parameters.batch_size, lower_bound, parameters.upper_bound,
                 parameters.l_norm, time_elapsed, accuracy
                    , parameters.file_name])
    except Exception as e:
        print("An exeption occured while writing to file")
        logging.exception(str(e) + "\n\n")
    finally:
        write_lock.release()


def reset_keras():
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    """
    try:
        del classifier # this is from global space - change this as you need
    except:
        pass"""

    print("clear gc", gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.Session(config=config))


def multithreadded_calculations(parameters):
    start_time = timer.time()

    setDynamicGPUAllocation()
    skip_architecture, accuracy = train_and_get_accuracy_of_nn(parameters.file_name,
                                                               parameters.filters,
                                                               parameters.kernels,
                                                               parameters.tf_activation,
                                                               parameters.has_batch_normalization)

    if not skip_architecture:
        skip_architecture, lower_bound = calculate_lower_bound(accuracy,
                                                               parameters.file_name,
                                                               parameters.num_image,
                                                               parameters.l_norm,
                                                               parameters.nn_architecture,
                                                               parameters.activation_function_string)

    reset_keras()
    if not skip_architecture:
        time_elapsed = timer.time() - start_time
        print("time elapsed", time_elapsed)

        write_to_file(parameters, lower_bound, accuracy, time_elapsed)


def pool_init(l1, l2):
    global write_lock
    global keras_lock
    write_lock = l1
    keras_lock = l2


def main():
    print("You have {} cores at your disposal.".format(multiprocessing.cpu_count()))

    l1 = multiprocessing.Lock()
    l2 = multiprocessing.Lock()
    pool = multiprocessing.Pool(initializer=pool_init, initargs=(l1, l2))

    make_result_file(CnnTestParameters.result_folder, CnnTestParameters.result_file)
    logging.basicConfig(filename='log.log', level="ERROR")
    for filter_size in range(2, 128, 8):
        for has_batch_normalization in [False]:
            for depth in range(1, 10, 2):
                for kernel_size in range(3, 15, 2):
                    for activation_function_string in ["ada", "sigmoid", "arctan", "tanh"]:
                        parameters = CnnTestParameters()
                        parameters.tf_activation = get_tf_activation_function_from_string(activation_function_string)
                        parameters.activation_function_string = activation_function_string
                        parameters.depth = depth
                        parameters.kernel_size = kernel_size
                        parameters.filter_size = filter_size
                        parameters.filters = [filter_size for i in range(depth)]
                        parameters.kernels = [kernel_size for i in range(depth)]
                        parameters.has_batch_normalization = has_batch_normalization

                        parameters.file_name = get_name(parameters)

                        pool_init(l1, l2)
                        #multithreadded_calculations(parameters)

                        pool.apply_async(multithreadded_calculations, (parameters,))
    pool.close()
    pool.join()


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


class CnnTestParameters:
    dataset = "mnist"
    nn_architecture = NnArchitecture.ONLY_CNN.value
    pooling = "None"
    epochs = 100
    stride = 1
    bias = True
    initializer = "glorot_uniform"
    regulizer = "None"
    temperature = 1
    batch_size = 128
    num_image = 10
    l_norm = "i"
    width = "null"
    upper_bound = None
    result_folder = 'results2/'
    result_file = 'results.csv'


if __name__ == "__main__":
    main()

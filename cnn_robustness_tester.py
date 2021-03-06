import sys
import traceback

from datasets.setup_GTSRB import GTSRB
from datasets.setup_calTech_101_silhouettes import CaltechSiluettes
from datasets.setup_cifar100 import CIFAR100
from datasets.setup_dogs_and_cats import DogsAndCats
from datasets.setup_sign_language import SignLanguage
from datasets.setup_rockpaperscissors import RockPaperScissors
from hyper_parameters import hyper_parameters

_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode('latin1'))

from CNN_Cert.train_cnn import train as train_cnn
from CNN_Cert.pymain import run_cnn
from datasets.setup_cifar import CIFAR
from datasets.setup_tinyimagenet import TinyImagenet
import csv
import os.path
from Attacks.cw_attack import cw_attack
import time as timer
from datasets.setup_mnist import MNIST
from enum import Enum
import logging
from datetime import datetime
import multiprocessing
import gc
import time
import os
from numba import cuda
import pandas as pd


class NnArchitecture(Enum):
    ONLY_CNN = "only_cnn"


def get_name(parameter_class):
    directory = "output/models"
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


def is_file_duplicated(file_name):
    models_meta_file = "output/models_meta.csv"
    models_meta = pd.read_csv(models_meta_file)
    file_name_column = "file_name"
    return len(models_meta[(models_meta[file_name_column] == file_name)]) > 1


def delete_file_and_rows_with_file_name(file_name):
    models_meta_file = "output/models_meta.csv"
    lower_bound_file = "output/results/results.csv"
    upper_bound_file = "upper_bound.csv"
    success_rate_file = "success_rate.csv"
    HSJA_upper_bound_file = "HSJA_upper_bound.csv"
    emprirical_robustness_file = "emprirical_robustness.csv"
    files = [models_meta_file, lower_bound_file, upper_bound_file, success_rate_file, HSJA_upper_bound_file, emprirical_robustness_file]
    try:
        os.remove(file_name)
    except:
        print("Could not delete file")
    for file in files:
        try:
            delete_duplicated_lines_in_file(file_name, file)
        except:
            None


def delete_duplicated_lines_in_file(file_name, file):
    file_name_column = "file_name"
    df = pd.read_csv(file)
    print(file, len(df))
    df = df[(df[file_name_column] != file_name)]
    df.to_csv(file, index=False)
    print(file, len(df))


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
                             "lower_bound", "upper_bound", "l_norm", "time_elapsed", "accuracy", "early_stoppping",
                             "padding_same", "Cnn-cert-core"
                                , "file_name"])


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
                 parameters.l_norm, time_elapsed, accuracy, parameters.use_early_stopping,
                 parameters.use_padding_same, parameters.use_cnnc_core
                    , parameters.file_name])
    except Exception as e:
        print("An exeption occured while writing to file")
        logging.exception(str(traceback.format_exc()) + "\n\n")
    finally:
        write_lock.release()


def make_upper_bound_file(file_name="upper_bound.csv"):
    if not os.path.exists(file_name):
        print("made new upper_bound_file")
        with open(file_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["time_elapsed", "upper_bound", "file_name", "l_norm"])


def fn(correct, predicted, tf):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)


def setDynamicGPUAllocation(tf):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)

    sess = tf.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras


def train_and_save_network(file_name, filters, kernels, epochs, tf_activation, batch_normalization, use_padding_same,
                           use_early_stopping, batch_size, _dataset_data, train_on_adversaries):
    train_cnn(_dataset_data,
              file_name=file_name,
              filters=filters,
              kernels=kernels,
              num_epochs=epochs,
              activation=tf_activation,
              bn=batch_normalization,
              batch_size=batch_size,
              use_padding_same=use_padding_same,
              use_early_stopping=use_early_stopping,
              train_on_adversaries=train_on_adversaries)


def get_lower_bound(file_name, num_image, l_norm, use_cnnc_core, activation_function_string, _dataset_data):
    avg_lower_bound, total_time = run_cnn(file_name,
                                          num_image,
                                          l_norm,
                                          _dataset_data,
                                          core=use_cnnc_core,
                                          activation=activation_function_string)
    return avg_lower_bound


def get_upper_bound_and_time(file_name, l_norm, num_image, sess, _dataset_data):
    upper_bound, time_taken = cw_attack(file_name, l_norm, sess, num_image=num_image, data_set_class=_dataset_data)
    return upper_bound, time_taken


def csv_contains_file(csv_file, file_name, parameters):
    with open(csv_file, "r") as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')

        first_row = next(csvreader)

        file_name_column = 0
        for i in range(len(first_row)):
            if first_row[i] == "file_name":
                file_name_column = i
                break

        cnnc_column = 0
        for i in range(len(first_row)):
            if first_row[i] == "Cnn-cert-core":
                cnnc_column = i
                break

        norm_column = 0
        for i in range(len(first_row)):
            if first_row[i] == "l_norm":
                norm_column = i
                break

        for row in csvreader:
            if row[file_name_column] == file_name:  # and row[cnnc_column] == parameters.use_cnnc_core:
                if str(row[cnnc_column]) == str(parameters.use_cnnc_core):
                    if str(row[norm_column]) == str(parameters.l_norm):
                        return True
    return False


def upper_bounds_csv_contains_file(csv_file, file_name, parameters):
    with open(csv_file, "r") as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')

        first_row = next(csvreader)

        file_name_column = 0
        for i in range(len(first_row)):
            if first_row[i] == "file_name":
                file_name_column = i
                break

        norm_column = 0
        for i in range(len(first_row)):
            if first_row[i] == "l_norm":
                norm_column = i
                break

        for row in csvreader:
            if row[file_name_column] == file_name:
                if str(row[norm_column]) == str(parameters.l_norm):
                    return True
    return False


def file_exists(file):
    return os.path.exists(file)


def get_tf_activation_function_from_string(activation_function_string, tf):
    if activation_function_string == "sigmoid":
        return tf.math.sigmoid
    if activation_function_string == "ada":
        return tf.nn.relu
    if activation_function_string == "tanh":
        return tf.math.tanh
    if activation_function_string == "arctan":
        return tf.math.atan


def calculate_lower_bound(file_name, num_image, l_norm, use_cnnc_core, activation_function_string, _dataset_data):
    return get_lower_bound(file_name,
                           num_image,
                           l_norm,
                           use_cnnc_core,
                           activation_function_string,
                           _dataset_data)

    """
    try:
        del classifier # this is from global space - change this as you need
    except:
        pass"""

    print("clear gc", gc.collect())  # if it's done something you should see a number being outputted


def get_dynamic_keras_config(tf):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    return config


def train_nn(parameters, file_name, filters, kernels, epochs, tf_activation, batch_normalization, use_padding_same,
             use_early_stopping, batch_size, _dataset_data, tf, train_on_adversaries):
    try:
        # reset_cuda()
        print(datetime.now())
        print(f"\ntraining with {parameter_string(parameters)}\n", flush=True)
        train_and_save_network(file_name,
                               filters,
                               kernels,
                               epochs,
                               tf_activation,
                               batch_normalization,
                               use_padding_same,
                               use_early_stopping,
                               batch_size,
                               _dataset_data=_dataset_data,
                               train_on_adversaries=train_on_adversaries)
        # reset_cuda()
        gc.collect()
    except Exception as e:
        print("Error: An exeption occured while training network", e)
        date = str(datetime.now())
        logging.exception("\n =================\n\n"
                          + date +
                          "\nThis file had an error: \n"
                          + file_name +
                          "\n" + str(traceback.format_exc()) +
                          "\n\n")


def reset_cuda():
    try:
        print("closing", cuda.get_current_device())
        cuda.close()
    except Exception as e:
        print("Could not reset cuda: ", traceback.format_exc())


def gpu_calculations(parameters):
    import tensorflow as tf
    try:
        parameters.tf_activation = get_tf_activation_function_from_string(
            parameters.activation_function_string, tf)

        if not file_exists(parameters.file_name):
            train_nn(parameters,
                     parameters.file_name,
                     parameters.filters,
                     parameters.kernels,
                     parameters.epochs,
                     parameters.tf_activation,
                     parameters.has_batch_normalization,
                     parameters.use_padding_same,
                     parameters.use_early_stopping,
                     parameters.batch_size,
                     dataset_data,
                     tf,
                     parameters.train_on_adversaries)
            print(f"\ndone training with {parameter_string(parameters)}\n", flush=True)
            return
        else:
            print("Neural network already created - {} - {}".format(datetime.now(), parameters.file_name), flush=True)
    finally:
        keras_lock.release()
        if not parameters.is_debugging:
            sys.exit(0)


def get_accuracy_of_nn_from_csv(csv_file, file_name):
    with open(csv_file, "r") as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')

        first_row = next(csvreader)
        for i in range(len(first_row)):
            if first_row[i] == "accuracy":
                accuracy_column = i

            elif first_row[i] == "file_name":
                name_column = i

        for row in csvreader:
            if row[name_column] == file_name:
                return row[accuracy_column]
    return None


def multithreadded_calculations(parameters):
    import tensorflow as tf
    semaphore.acquire()
    try:
        print(f"\nCalculating robustness of {parameter_string(parameters)}\n", flush=True)

        parameters.tf_activation = get_tf_activation_function_from_string(
            parameters.activation_function_string, tf)

        if not file_exists(parameters.file_name):
            print("File does not exist {}".format(parameters.file_name), flush=True)
            print_parameters(parameters)
            logging.exception("\n =================\n\n"
                              + str(datetime.now()) +
                              "\nThe error was here: \n"
                              + parameter_string(parameters) +
                              "\n" + "File does not exist {}".format(parameters.file_name) +
                              "\n\n")
            return

        start_time = timer.time()

        make_result_file(parameters.result_folder, parameters.result_file)

        debugprint(parameters.is_debugging, "reading results csv")
        if csv_contains_file(parameters.result_folder + parameters.result_file, parameters.file_name, parameters):
            print("Bounds already calculated for {}".format(parameters.file_name), flush=True)
            print_parameters(parameters)
            return

        debugprint(parameters.is_debugging, "reading models_meta.csv")
        accuracy = None#get_accuracy_of_nn_from_csv("output/models_meta.csv", parameters.file_name)

        """
        if float(accuracy) < 0.95:
            print("skiped", parameters.file_name, "as the accuracy was too low", flush=True)
            print_parameters(parameters)
            return
        """

        debugprint(parameters.is_debugging, "calculating lower bound")
        gpu_options = tf.GPUOptions(visible_device_list=_b("").decode('utf-8'))
        session_config = tf.ConfigProto(device_count={'GPU': 0}, gpu_options=gpu_options)
        sess = tf.Session(config=session_config)
        with sess.as_default():
            lower_bound = calculate_lower_bound(parameters.file_name,
                                                parameters.num_image,
                                                parameters.l_norm,
                                                parameters.use_cnnc_core,
                                                parameters.activation_function_string,
                                                dataset_data)

        time_elapsed = timer.time() - start_time

        debugprint(parameters.is_debugging, "writing to file")
        write_to_file(parameters, lower_bound, accuracy, time_elapsed)

        print("wrote to file", flush=True)
        print_parameters(parameters)

        return
    except Exception as e:
        print("Error: An exeption occured while calculating robustness", e)
        logging.exception("\n =================\n\n"
                          + str(datetime.now()) +
                          "\nThe error was here: \n"
                          + parameter_string(parameters) +
                          "\n" + str(traceback.format_exc()) +
                          "\n\n")
    finally:
        semaphore.release()
        # reset_cuda()
        gc.collect()


def upper_bound_calculations(parameters):
    keras_lock2.acquire()
    try:
        import tensorflow as tf
        print(f"\nCalculating upper bound of {parameter_string(parameters)}\n", flush=True)

        parameters.tf_activation = get_tf_activation_function_from_string(
            parameters.activation_function_string, tf)

        debugprint(parameters.is_debugging, "checking if model file exists")
        if not file_exists(parameters.file_name, use_cache=False):
            print("File does not exist {}".format(parameters.file_name), flush=True)
            print_parameters(parameters)
            logging.exception("\n =================\n\n"
                              + str(datetime.now()) +
                              "\nThe error was here: \n"
                              + parameter_string(parameters) +
                              "\n" + "File does not exist {}".format(parameters.file_name) +
                              "\n\n")
            return

        csv_name = parameters.upper_bound_result_file
        make_upper_bound_file(csv_name)

        add_l_norm_to_upper_bound_file()
        debugprint(parameters.is_debugging, "reading results csv")
        if upper_bounds_csv_contains_file(csv_name, parameters.file_name, parameters):
            print("Upper bounds already calculated for {}".format(parameters.file_name), flush=True)
            print_parameters(parameters)
            return

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = False
        sess = tf.Session(config=config)

        with sess.as_default():
            upper_bound, time_spent = get_upper_bound_and_time(parameters.file_name,
                                                               parameters.l_norm,
                                                               parameters.num_image,
                                                               sess,
                                                               dataset_data)

        debugprint(parameters.is_debugging, "writing upper bound to file")
        write_to_upper_bound_file(parameters, upper_bound, time_spent, csv_name)

        print("wrote upper bound to file", flush=True)
        print_parameters(parameters)

        return
    except Exception as e:
        print("Error: An exeption occured while calculating upper bound", e)
        logging.exception("\n =================\n\n"
                          + str(datetime.now()) +
                          "\nThe error was here: \n"
                          + parameter_string(parameters) +
                          "\n" + str(traceback.format_exc()) +
                          "\n\n")
    finally:
        # reset_cuda()
        keras_lock2.release()
        gc.collect()


def pool_init(l1, l2, l3, sema, data):
    global write_lock
    global keras_lock
    global keras_lock2
    global semaphore
    global dataset_data
    semaphore = sema
    write_lock = l1
    keras_lock = l2
    keras_lock2 = l3
    dataset_data = data


def parameter_string(parameters):
    return "depth={} filter_size={} kernel_size={} ac={} es={} pad={} cnnc={}" \
        .format(parameters.depth,
                parameters.filter_size,
                parameters.kernel_size,
                parameters.activation_function_string,
                parameters.use_early_stopping,
                parameters.use_padding_same,
                parameters.use_cnnc_core)


def print_parameters(parameters):
    print()
    print(parameter_string(parameters))
    print(datetime.now())
    print(
        "========================================================================================")
    print("", flush=True)


def debugprint(is_debugging, text):
    if is_debugging:
        print(text, flush=True)


def get_data(dataset):
    if dataset == "mnist":
        data = MNIST()
    elif dataset == "cifar":
        data = CIFAR()
    elif dataset == "tinyImagenet":
        data = TinyImagenet()
    elif dataset == "caltechSilhouettes":
        data = CaltechSiluettes()
    elif dataset == "GTSRB":
        data = GTSRB()
    elif dataset == "cifar100":
        data = CIFAR100()
    elif dataset == "dogs-and-cats":
        data = DogsAndCats()
    elif dataset == "sign-language":
        data = SignLanguage()
    elif dataset == "rockpaperscissors":
        data = RockPaperScissors()
    else:
        raise NameError(f"{dataset} is not a valid dataset")
    return data


def main():
    print("args: ", sys.argv)
    _, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8 = sys.argv
    cpu = arg1 == "cpu" or arg2 == "cpu"
    gpu = arg1 == "gpu" or arg2 == "gpu"
    debugging = arg3 == "debugging"
    path = arg4
    dataset = arg5
    upper_bound = arg6 == "upper"
    num_cpus = int(arg7)
    train_on_adversaries = arg8 == "adv"

    print("cpu:", cpu)
    print("gpu:", gpu)
    print("debugging:", debugging)
    print("path:", path)
    print("dataset:", dataset)
    print("upper_bound:", upper_bound)
    print("arg7:", arg7)

    set_path(path)

    set_path(dataset)

    try:
        model_files = os.listdir("output/models")
    except:
        model_files = []

    if not debugging:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print("You have {} cores at your disposal.".format(multiprocessing.cpu_count()))
    print(f"The path is {path}")

    if debugging:
        print(f"cpu: {cpu}")
        print(f"gpu: {gpu}")
        print(f"path: {path}/")

    processes = min(multiprocessing.cpu_count(), num_cpus)

    data = get_data(dataset)

    l1 = multiprocessing.Lock()
    l2 = multiprocessing.Lock()
    l3 = multiprocessing.Lock()
    sema = multiprocessing.Semaphore(processes)

    cpu_pool = multiprocessing.Pool(processes, initializer=pool_init, initargs=(l1, l2, l3, sema, data), maxtasksperchild=1)
    gpu_pool = multiprocessing.Pool(1, initializer=pool_init, initargs=(l1, l2, l3, sema, data), maxtasksperchild=1)

    pool_init(l1, l2, l3, sema, data)

    logging.basicConfig(filename='log.log', level="ERROR")

    for parameters in hyper_parameters(dataset,
                                       model_files,
                                       debugging,
                                       train_on_adversaries):
        #if is_file_duplicated(parameters.file_name):
            #delete_file_and_rows_with_file_name(parameters.file_name)
        if debugging:
            print(parameters.is_debugging)
            if gpu:
                keras_lock.acquire()
                gpu_calculations(parameters)
            if cpu:
                print("hhh")
                multithreadded_calculations(parameters)

        else:
            if gpu:
                keras_lock.acquire()
                gpu_process = multiprocessing.Process(target=gpu_calculations, args=(parameters,))
                gpu_process.start()

                keras_lock.acquire()
                gpu_process.terminate()
                keras_lock.release()

                gc.collect()
            if cpu:
                cpu_pool.apply_async(multithreadded_calculations, (parameters,))

    print("Waiting for processes to finish")
    gpu_pool.close()
    cpu_pool.close()
    gpu_pool.join()
    cpu_pool.join()
    print("program finished")


def set_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)


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


if __name__ == "__main__":
    main()

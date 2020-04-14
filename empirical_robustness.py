import csv
import logging
import os
import sys
from datetime import datetime

import tensorflow as tf

from Attacks.PGD_wrapper import get_accuracy as get_PGD_accuracy
from cnn_robustness_tester import get_tf_activation_function_from_string, debugprint, file_exists, print_parameters, parameter_string, set_path, get_data
from hyper_parameters import hyper_parameters


def make_result_file(file_name="empirical_robustness.csv"):
    if not os.path.exists(file_name):
        print("made new upper_bound_file")
        with open(file_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["time_elapsed", "accuracy", "epsilon", "file_name"])


def write_to_file(parameters, accuracy, time_elapsed, csv_file):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [time_elapsed, accuracy, str(parameters.epsilon), parameters.file_name])


def csv_contains_file(csv_file, file_name, parameters):
    with open(csv_file, "r") as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')

        first_row = next(csvreader)

        file_name_column = 0
        for i in range(len(first_row)):
            if first_row[i] == "file_name":
                file_name_column = i
                break

        epsilon_column = 0
        for i in range(len(first_row)):
            if first_row[i] == "epsilon":
                epsilon_column = i
                break

        for row in csvreader:
            if row[file_name_column] == file_name:
                if str(row[epsilon_column]) == str(parameters.epsilon):
                    return True
    return False

def empirical_robustness_calculations(parameters):
    parameters.tf_activation = get_tf_activation_function_from_string(
        parameters.activation_function_string, tf)

    debugprint(parameters.isDebugging, "checking if model file exists")
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

    csv_name = "emprirical_robustness.csv"
    make_result_file(csv_name)

    debugprint(parameters.isDebugging, "reading results csv")
    if csv_contains_file(csv_name, parameters.file_name, parameters):
        print(f"Empirical bounds already calculated for {parameters.file_name} and {parameters.epsilon}", flush=True)
        print_parameters(parameters)
        return

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False
    sess = tf.Session(config=config)

    with sess.as_default():
        accuracy, time_spent = get_PGD_accuracy(parameters.file_name,
                                                   sess,
                                                   parameters.epsilon,
                                                   parameters.steps,
                                                   parameters.step_size,
                                                   dataset_data)

    debugprint(parameters.isDebugging, "writing to file")
    write_to_file(parameters, accuracy, time_spent, csv_name)

    print("wrote to file", flush=True)
    print_parameters(parameters)

    return


def init_globals(data):
    global dataset_data
    dataset_data = data

def main():
    _, path, dataset = sys.argv
    set_path(path)
    set_path(dataset)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        model_files = os.listdir("output/models")
    except:
        model_files = []

    data = get_data(dataset)
    init_globals(data)

    epsilons = [0.03, 0.01, 0.005]
    for parameters in hyper_parameters(dataset=dataset,
                                       model_files=model_files):
        parameters.steps = 1000
        parameters.step_size = 0.001
        for epsilon in epsilons:
            parameters.epsilon = epsilon
            empirical_robustness_calculations(parameters)


if __name__ == "__main__":
    main()

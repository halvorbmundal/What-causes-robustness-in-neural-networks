import csv
import gc
import logging
import os
import sys
from datetime import datetime

import tensorflow as tf

from Attacks.MIM_wrapper import get_accuracy as get_MIM_accuracy
from Attacks.PGD_wrapper import get_accuracy as get_PDG_accuracy
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

    debugprint(parameters.is_debugging, "checking if model file exists")
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

    if parameters.success_rate_attack == "mim":
        csv_name = "success_rate.csv"
        attack = get_MIM_accuracy
    elif parameters.success_rate_attack == "pdg":
        csv_name = "emprirical_robustness.csv"
        attack = get_PDG_accuracy
    make_result_file(csv_name)

    debugprint(parameters.is_debugging, "reading results csv")
    if csv_contains_file(csv_name, parameters.file_name, parameters):
        print(f"Empirical bounds already calculated for {parameters.file_name} and {parameters.epsilon}", flush=True)
        print_parameters(parameters)
        return

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    #config.log_device_placement = False
    sess = tf.Session()#(config=config)

    with sess.as_default():
        accuracy, time_spent = attack(parameters.file_name,
                                                   sess,
                                                   parameters.epsilon,
                                                   parameters.steps,
                                                   parameters.step_size,
                                                   dataset_data)

    debugprint(parameters.is_debugging, "writing to file")
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
    print(path, dataset, flush=True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        model_files = os.listdir("output/models")
    except:
        model_files = []

    data = get_data(dataset)
    init_globals(data)

    params = hyper_parameters(dataset=dataset, model_files=model_files)

    epsilons = get_epsilon(dataset)
    for success_rate_attack in ["mim", "pdg"]:
        for epsilon in epsilons:
            for parameters in params:
                parameters.steps = 1000
                parameters.step_size = 0.001
                parameters.epsilon = epsilon
                parameters.success_rate_attack = success_rate_attack

                empirical_robustness_calculations(parameters)
                gc.collect()


def get_epsilon(dataset):
    if dataset == "mnist":
        epsilons = [0.05]
    elif dataset == "cifar":
        epsilons = [0.004]
    elif dataset == "caltechSilhouettes":
        epsilons = [0.04]
    elif dataset == "GTSRB":
        epsilons = [0.015]
    elif dataset == "sign-language":
        epsilons = [0.013]
    elif dataset == "rockpaperscissors":
        epsilons = [0.02]
    return epsilons


if __name__ == "__main__":
    main()

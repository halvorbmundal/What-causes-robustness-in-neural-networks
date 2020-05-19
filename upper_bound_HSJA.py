import csv
import gc
import logging
import os
import sys
from datetime import datetime

import tensorflow as tf

from Attacks.HSJA_wrapper import hsja_attack
from cnn_robustness_tester import get_tf_activation_function_from_string, debugprint, file_exists, print_parameters, parameter_string, set_path, get_data, \
    make_upper_bound_file, upper_bounds_csv_contains_file
from hyper_parameters import hyper_parameters

"""
def add_l_norm_to_upper_bound_file(file_name="upper_bound.csv"):
    csv_input = pd.read_csv(file_name)
    if "l_norm" in csv_input.keys():
        return
    csv_input['l_norm'] = "i"
    csv_input.to_csv(file_name, index=False)"""


def write_to_upper_bound_file(parameters, upper_bound, time_elapsed, csv_file):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [time_elapsed, upper_bound, parameters.file_name, parameters.l_norm])


def upper_bound_calculations(parameters):
    if parameters.l_norm == "1":
        return

    print(f"\nCalculating upper bound of {parameter_string(parameters)}\n", flush=True)

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

    csv_name = "HSJA_" + parameters.upper_bound_result_file
    make_upper_bound_file(csv_name)

    debugprint(parameters.is_debugging, "reading results csv")
    if upper_bounds_csv_contains_file(csv_name, parameters.file_name, parameters):
        print("Upper bounds already calculated for {}".format(parameters.file_name), flush=True)
        print_parameters(parameters)
        return

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False
    sess = tf.Session(config=config)

    try:
        with sess.as_default():
            upper_bound, time_spent = hsja_attack(parameters.file_name,
                                                  parameters.l_norm,
                                                  sess,
                                                  parameters.num_image,
                                                  dataset_data)
        if time_spent == 0:
            return
    except:
        return

    debugprint(parameters.is_debugging, "writing upper bound to file")
    write_to_upper_bound_file(parameters, upper_bound, time_spent, csv_name)

    print("wrote upper bound to file", flush=True)
    print_parameters(parameters)

    gc.collect()
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

    for parameters in params:
        upper_bound_calculations(parameters)

        gc.collect()


if __name__ == "__main__":
    main()

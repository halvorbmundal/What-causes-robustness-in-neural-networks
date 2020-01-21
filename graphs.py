import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from halvor import get_lower_bound
import csv


def display_graph(data, name, addarg=None, folder_name=""):
    x = []
    avg_lower, std_lower = [], []
    df = data
    if addarg is not None:
        df = df[addarg]
    unique = df[name].unique()
    unique.sort()
    for i in unique:
        has_value_i = (df[name] == i)
        dff = df[has_value_i]

        if i == "ada":
            x.append("relu")
        else:
            x.append(i)

        avg_lower.append(dff["lower_bound"].mean())
        std_lower.append(dff["lower_bound"].std())

    plt.errorbar(x, avg_lower, std_lower)
    if name == "width":
        plt.xscale("log", basex=2)
    try:
        os.mkdir("graphs/" + folder_name)
        None
    except:
        None

    #plt.title(name + " " + folder_name)
    plt.xlabel(name)
    plt.ylabel("Robustness")

    plt.savefig("graphs/" +folder_name+"/" + " " + name)

    plt.show()


def hmmm(data, name, addarg=None, folder_name=""):
    x = []
    avg1 = []
    std1 = []
    avg2 = []
    std2 = []
    df = data
    if addarg is not None:
        df = df[addarg]
    unique = df[name].unique()
    unique.sort()
    for i in unique:
        has_value_i = (df[name] == i)

        try:
            dff = df[has_value_i][df["upper_bound"]].notnull()
        except:
            dff = df[has_value_i]
        dff = df[has_value_i]

        if i == "ada":
            x.append("relu")
        else:
            x.append(i)

        avg1.append(dff["lower_bound"].mean())
        std1.append(dff["lower_bound"].std())

        avg2.append(dff["upper_bound"].mean())
        std2.append(dff["upper_bound"].std())
        
    plt.errorbar(x, avg2, std2)
    plt.errorbar(x, avg1, std1)
    if name == "width":
        plt.xscale("log", basex=2)
    try:
        os.mkdir("graphs/" + folder_name)
        None
    except:
        None

    plt.title(name) #+ " " + folder_name)

    plt.savefig("graphs/" +folder_name+"/" + " " + name)

    plt.show()


def hmm2(data, params=None, name=""):
    if params is None:
        hmmm(data, "depth", folder_name=name)
        hmmm(data, "width", folder_name=name)
        hmmm(data, "filter", folder_name=name)
        hmmm(data, "kernel", folder_name=name)
    else:
        hmmm(data, "depth", addarg=params, folder_name=name)
        hmmm(data, "width", addarg=params, folder_name=name)
        hmmm(data, "filter", addarg=params, folder_name=name)
        hmmm(data, "kernel", addarg=params, folder_name=name)


def display_all_variables(data, params=None, name=""):
    if params is None:
        display_graph(data, "depth", folder_name=name)
        #display_graph(data, "width", folder_name=name)
        display_graph(data, "filter", folder_name=name)
        display_graph(data, "kernel", folder_name=name)
        display_graph(data, "activation_function", folder_name=name)
    else:
        display_graph(data, "depth", addarg=params, folder_name=name)
        #display_graph(data, "width", addarg=params, folder_name=name)
        display_graph(data, "filter", addarg=params, folder_name=name)
        display_graph(data, "kernel", addarg=params, folder_name=name)
        display_graph(data, "activation_function", addarg=params, folder_name=name)


def combine_csvs():
    d1_3 = pd.read_csv('results/results_10_sapmles.csv')
    d4 = pd.read_csv('results/results_10_d4_sapmles.csv')
    d5 = pd.read_csv('results/results_10_d_5_8_sapmles.csv')
    d6 = pd.read_csv('results/results_10_d6_sapmles.csv')
    d7 = pd.read_csv('results/results_10_d7_sapmles.csv')
    d8 = pd.read_csv('results/results_10_d_8_9_sapmles.csv')
    d9 = pd.read_csv('results/results_10_d_9_sapmles.csv')

    frames = [d1_3, d4, d5, d6, d7, d8, d9]

    data = pd.concat(frames).drop_duplicates(subset="file_name")

    return data

def test_robustness(r_list, start=0):
    result_file = "results/best_networks_100_large.csv"
    """
    with open(result_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["activation_function", "inital_lower_bound", "lower_bound", "file_name"])"""

    i = 0
    for index, row in r_list.iterrows():
        if i < start:
            i += 1
            continue
        file_name = row["file_name"]
        l_norm = row["l_norm"]
        nn_architecture = row["nn_architecture"]
        activation_function = row["activation_function"]
        inital_lower_bound = row["lower_bound"]
        print(index, "   ", inital_lower_bound)
        lower_bound = get_lower_bound(file_name, 100, l_norm, nn_architecture, activation_function)
        with open(result_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([activation_function, inital_lower_bound, lower_bound, file_name])

def main():
    #data = combine_csvs()
    data2 = pd.read_csv('results/results_large_filters.csv')
    data = pd.read_csv('results/results_10_combined_3.csv')
    #data = pd.read_csv('results/results_10_combined_3.csv')
    #data.to_csv('results/results_10_combined_5.csv', index=False)
    #data = data[(data["activation_function"] == "ada")]
    print(data.nlargest(10, "lower_bound")["lower_bound"])
    #test_robustness(data.nlargest(100, "lower_bound"), 1)
    print("smaples", len(data.index))
    print(data["activation_function"].value_counts())
    # data = data.dropna(subset =["accuracy"])
    # print("smaples", len(data.index))
    # data = data[(data["accuracy"] >= 0.95)]
    # print("smaples > 0.95 accuracy", len(data.index))

    #display_all_variables(data, name="initial examination")
    display_all_variables(data2, name="initial examination")


    kernel_filter = (data["kernel"] <= 7) & (data["kernel"] >= 5)
    # display_all_variables(data, kernel_filter, "kernel5,7")

    depth_filter = (data["depth"] >= 3)
    # display_all_variables(data, kernel_filter & depth_filter, "depth5-")

    # hmm2(data, (data["depth"] == 6) , "filter7-depth6")

    params3 = (data["kernel"] < 9) & (data["kernel"] > 4) & (data["depth"] > 3)
    # hmm2(data, params3, "k5-7.d3-")


main()
"""
print(data.columns)
ax = plt.gca()
data.plot(kind='scatter', x='depth', y='upper_bound', color='orange', ax=ax)
data.plot(kind='scatter', x='depth', y='lower_bound', color='blue', ax=ax)
plt.title("depth")
plt.show()

ax = plt.gca()
data.plot(kind='scatter', x='width', y='upper_bound', color='orange', ax=ax)
data.plot(kind='scatter', x='width', y='lower_bound', color='blue', ax=ax)
plt.title("width")
plt.show()

ax = plt.gca()
data.plot(kind='scatter', x='accuracy', y='upper_bound', color='orange', ax=ax)
data.plot(kind='scatter', x='accuracy', y='lower_bound', color='blue', ax=ax)
plt.title("accuracy")
plt.show()

filter = [x for x in data["filter"].values.tolist()]
plt.plot(filter, data["lower_bound"].values.tolist(), 'o')
plt.plot(filter, data["upper_bound"].values.tolist(), 'o')
plt.title("filter_size")
plt.show()

kernel = [x for x in data["kernel"].values.tolist()]
plt.plot(kernel, data["lower_bound"].values.tolist(), 'o')
plt.plot(kernel, data["upper_bound"].values.tolist(), 'o')
plt.title("kernel_size")
plt.show()



#plt.errorbar()"""

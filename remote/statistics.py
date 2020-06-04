import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
import statsmodels.api as sm
import json

import scipy.optimize as soptimize

from sklearn.datasets import load_iris
from sklearn import tree

_dataset = "dataset"
_file_name = "file_name"
_kernel = "kernel"
_depth = "depth"
_accuracy = "accuracy"
_filter = "filter"
_activation_function = "activation_function"
_early_stoppping = "early_stoppping"
_padding_same = "padding_same"
_batch_normalization = "has_batch_normalization"
_l_norm = "l_norm"
_upper_bound, _lower_bound = "upper_bound", "lower_bound"

"""
def get_df(columns, path, norm):
    results = pd.read_csv(path + 'results/results.csv')
    accuracy = pd.read_csv(path + 'models_meta.csv')
    ub_df: pd.DataFrame = get_upper_bound_df(path)

    df: pd.DataFrame = results.drop(["upper_bound"], axis=1)
    data_df = pd.merge(left=df, right=ub_df, left_on=['file_name', _l_norm], right_on=['file_name', _l_norm],
                       suffixes=("_x", ""))
    # apply queries
    data_df = data_df[queries(data_df, norm)]

    results = results[~results[[_file_name, "Cnn-cert-core", "l_norm"]].duplicated(keep="first")]

    #  Merge with
    data_df: pd.DataFrame = pd.merge(left=data_df, right=accuracy, left_on='file_name', right_on='file_name',
                                     suffixes=("_x", ""))

    df = data_df.iloc[0:, [data_df.columns.get_loc(c) for c in columns]]
    if _activation_function in columns:
        df_with_dummies = pd.get_dummies(df, columns=[_activation_function], prefix="", drop_first=True)
    else:
        df_with_dummies = df

    return df_with_dummies, data_df["lower_bound"], data_df["accuracy"], data_df["time_per_epoch"], data_df["best_epoch"], data_df[_activation_function], \
           data_df["upper_bound"]"""


def get_numppy_arrays(df):
    x = df.to_numpy()
    x_individual = {}
    for i in df.keys():
        x_individual[i] = df[i].to_numpy()
    return x, x_individual


def linear_regression(X, y):
    if len(X) == 0:
        raise ValueError("Data set is empty")
    X2 = sm.add_constant(X)
    model = sm.OLS(y, X2)
    est2 = model.fit()
    return est2


def plot_single_variable(X, y, name):
    plt.scatter(X, y)

    plt.xlabel(name)
    plt.ylabel("Robustness")
    plt.show()


"""
def get_y(df):
    return df["lower_bound"]


def plot_3d(x, y, z, c):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # ax.plot3D(x, y, z, 'gray')
    ax.scatter3D(x, y, z, cmap='hsv');

    plt.show()


def pca(X, y):
    x = StandardScaler().fit_transform(X)
    pca = PCA(2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, df[['target']]], axis=1)


def decicion_tree(X, y, columns=None):
    clf = tree.DecisionTreeRegressor(random_state=0, max_depth=4)
    clf = clf.fit(X, y)
    plt.figure(figsize=(7, 4), dpi=250)
    tree.plot_tree(clf, feature_names=columns)
    # tree.export_graphviz(clf)
    plt.show()"""


def error_plot_column(df, column_name, query=None, y="lower_bound"):
    plt.xlabel(column_name)
    plt.ylabel(y)
    x = []
    avg_lower, std_lower = [], []

    if query is not None:
        df = df[query]
    unique = df[column_name].unique()
    unique.sort()
    for i in unique:
        has_value_i = (df[column_name] == i)
        dff = df[has_value_i]

        if i == "ada":
            x.append("relu")
        else:
            x.append(i)

        avg_lower.append(dff[y].mean())
        std_lower.append(dff[y].std())

    plt.errorbar(x, avg_lower, std_lower, label=y)


# def err_plot_query():
#    (data["depth"] == 6)

def get_result_df(path):
    try:
        results = pd.read_csv(path + 'results.csv')
    except:
        results = pd.read_csv("../v10/" + path + 'results.csv')
    return results


def poly_reg(X, y, p=2):
    model = LinearRegression()
    poly = PolynomialFeatures(degree=p)
    poly_X = poly.fit_transform(X, y)
    est = sm.OLS(y, poly_X)
    est2 = est.fit()
    return poly, est2

    """
    model.fit(poly_X, y)
    return poly, model"""


def show_error_bars(df, x, query, ys, title, fig):
    for i in ys:
        error_plot_column(df, x, query, y=i)
    y = df[ys[0]]
    fig.set_ylim(bottom=0)
    fig.set_title(title)
    # plt.show()


def plot_column(all_queries, column, dataset, df, ys):
    error_plot_column(df, column, query=all_queries, y=ys[1])
    plt.title(dataset)
    plt.show()


def get_merged_adversarial_error_df(csv_path, lower_bounds):
    df: pd.DataFrame = pd.read_csv(csv_path)
    lower_bounds = lower_bounds[(lower_bounds[_l_norm] == "i")]
    df = pd.merge(left=lower_bounds, right=df, left_on=['file_name'], right_on=['file_name'],
                  suffixes=("_nautral", "_adversarial"))
    df["adversarial_error"] = df["accuracy_nautral"].to_numpy() - df["accuracy_adversarial"].to_numpy()
    df["sr"] = df["accuracy_adversarial"].to_numpy() / df["accuracy_nautral"].to_numpy()
    return df["sr"], df


def get_merged_adversarial_train_df(path, lower_bounds, norm):
    assert norm == "i", "Norm should be \"i\"."
    adv_lower_bound: pd.DataFrame = pd.read_csv(f"{path}adversarial_results.csv")
    adv_lower_bound["file_name"] = adv_lower_bound["file_name"].apply(lambda x: x[11:])
    lower_bounds["file_name"] = lower_bounds["file_name"].apply(lambda x: x[len("output/models/"):])
    natural_accuracy: pd.DataFrame = pd.read_csv(f"{path}adv_model_natural_accuracy.csv")

    adv_models_meta: pd.DataFrame = pd.read_csv(f"{path}adversarial_models_meta.csv")
    adv_models_meta["file_name"] = adv_models_meta["file_name"].apply(lambda x: x[11:])
    adv_lower_bound = adv_lower_bound.drop(columns=["accuracy"])
    adv_lower_bound: pd.DataFrame = pd.merge(left=adv_lower_bound, right=adv_models_meta, left_on=['file_name'], right_on=['file_name'])

    df: pd.DataFrame = pd.merge(left=natural_accuracy, right=adv_lower_bound, left_on=['file_name'], right_on=['file_name'],
                                suffixes=("", "_adversarial"))

    df = df[queries(df, norm)]

    df["file_name"] = df["file_name"].apply(lambda x: x[:-4])
    df: pd.DataFrame = pd.merge(left=lower_bounds[["accuracy", "lower_bound", "file_name"]], right=df, left_on=['file_name'], right_on=['file_name'],
                                suffixes=("", "_advtrained"))

    print_comparison(df)

    return df["lower_bound_advtrained"], df


def print_comparison(df):
    mean_nat_lb = np.mean(df["lower_bound"])
    mean_adv_lb = np.mean(df["lower_bound_advtrained"])
    mean_acc = np.mean(df["accuracy"])
    mean_acc_advtrained = np.mean(df["accuracy_advtrained"])
    mean_adv_acc_advtrained = np.mean(df["accuracy_adversarial"])
    lb_incr = 100 * np.mean(df["lower_bound_advtrained"] / np.mean(df["lower_bound"]))
    print("Dataset & Natural lower bound & Robustness increase & Average accuracy")
    print(f"& {format(mean_nat_lb, '.4f')} & {format(mean_acc*100, '.1f')}\\%")

    print("Dataset & Adversarial & Robustness & Average  & Average adversarial")
    print(" & lower bound & increase &  accuracy & accuracy")
    print(f" & {format(mean_adv_lb, '.4f')} & {format(lb_incr, '.0f')}\\% &"
          f" {format(mean_acc_advtrained*100, '.1f')}\\% & {format(mean_adv_acc_advtrained*100, '.1f')}\\%")


def get_merged_upper_bound_df(csv_path, lower_bounds, norm):
    lower_bounds = lower_bounds.drop(["upper_bound"], axis=1)
    ub_df: pd.DataFrame = pd.read_csv(csv_path)
    df: pd.DataFrame = pd.merge(left=lower_bounds, right=ub_df, left_on=['file_name', _l_norm], right_on=['file_name', _l_norm],
                                suffixes=("_x", ""))
    df = df[queries(df, norm)]
    df = df.dropna(subset=["upper_bound", ])
    return df["upper_bound"], df


def set_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)


def normalize_0_to_1(x):
    return (x - min(x)) / (max(x) - min(x))


def exponential(x, a, k, b):
    return a * np.exp(x * k) + b


def display_line_on_scatter(X, y, results, columns, name, dataset):
    y = y.values
    if len(columns) == 4:
        print(columns)
        accuracy = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        plt.xlabel(columns[0])
        plt.ylabel("Robustness")
        if False:
            coefs = results.params
            y_relu = coefs[0] + coefs[1] * accuracy
            y_arctan = coefs[0] + coefs[1] * accuracy + coefs[2]
            y_sigmoid = coefs[0] + coefs[1] * accuracy + coefs[3]
            y_tanh = coefs[0] + coefs[1] * accuracy + coefs[4]

            plt.plot(accuracy, y_relu, label="relu", color="r")
            plt.plot(accuracy, y_arctan, label="arctan", color="g")
            plt.plot(accuracy, y_sigmoid, label="sigmoid", color="m")
            plt.plot(accuracy, y_tanh, label="tanh", color="blue")

        x_relu, x_arctan, x_sigmoid, x_tanh = [], [], [], []
        y_relu, y_arctan, y_sigmoid, y_tanh = [], [], [], []

        for i in range(len(X)):
            if X[i][1] == 1:
                x_arctan.append(X[i][0])
                y_arctan.append(y[i])
                plt.scatter(X[i][0], y[i], s=3, color="b", marker='o')
            elif X[i][2] == 1:
                x_sigmoid.append(X[i][0])
                y_sigmoid.append(y[i])
                plt.scatter(X[i][0], y[i], s=3, color="b", marker='o')
            elif X[i][3] == 1:
                x_tanh.append(X[i][0])
                y_tanh.append(y[i])
                plt.scatter(X[i][0], y[i], s=3, color="b", marker='o')
            else:
                x_relu.append(X[i][0])
                y_relu.append(y[i])
                plt.scatter(X[i][0], y[i], s=3, color="b", marker='o')

        legend_elements = [Line2D([0], [0], marker='o', color='w', label='arctan',
                                  markerfacecolor='g', markersize=6),
                           Line2D([0], [0], marker='o', color='w', label='sigmoid',
                                  markerfacecolor='m', markersize=6),
                           Line2D([0], [0], marker='o', color='w', label='tanh',
                                  markerfacecolor='blue', markersize=6),
                           Line2D([0], [0], marker='o', color='w', label='relu',
                                  markerfacecolor='r', markersize=6)
                           ]

        if False:
            fit_to_curve(x_relu, y_relu, label="relu", color="r")
            fit_to_curve(x_arctan, y_arctan, label="atan", color="g")
            fit_to_curve(x_sigmoid, y_sigmoid, label="sigmoid", color="m")
            fit_to_curve(x_tanh, y_tanh, label="tanh", color="blue")
            fit_to_curve(X[:, 0], y, label="tanh", color="black")

    elif len(columns) == 1:
        print(X[0])
        plt.scatter(X, y)
        x = np.linspace(np.min(X), np.max(X), 10)
        coef_const, coef_variable = results.params
        y = coef_const + coef_variable * x
        print(results.bse)
        std_const, std_variable = results.bse
        # plt.errorbar(x, y, std_variable, color="r")
    else:
        return
    plt.title(dataset + ", " + name)
    # plt.legend(handles=legend_elements)
    # plt.show()


def fit_to_curve(X, y, label, color):
    try:
        x_space = np.linspace(np.min(X), np.max(X), 100)
        popt_exponential, pcov_exponential = soptimize.curve_fit(exponential, X, y, maxfev=100000)
        plt.plot(x_space, popt_exponential[0] * np.exp(x_space * popt_exponential[1]) + popt_exponential[2], label=label, color=color)
    except Exception as e:
        print(e)


def plot_linear_aaccuracy_vs_robustness(coefs, accuracy, label, color):
    y_relu = coefs[0] + coefs[1] * accuracy
    plt.plot(accuracy, y_relu, label=label, color=color)


def do_linear_regression(x_df, y, columns, name, dataset, polynomial=None):
    x_df = x_df.iloc[0:, [x_df.columns.get_loc(c) for c in columns]]
    if _activation_function in x_df.keys():
        x_df = pd.get_dummies(x_df, columns=[_activation_function], prefix="", drop_first=True)

    X = x_df.to_numpy()
    print(len(x_df))
    print("min", np.min(y))
    print("mean", np.mean(y))
    print("median", np.median(y))
    print("max", np.max(y))

    columns = x_df.keys()

    # X, input_dict = get_numppy_arrays(x_df)

    if polynomial is not None:
        poly_features, results = poly_reg(X, y, polynomial)
        xname = poly_features.get_feature_names(x_df.columns)
    else:
        results = linear_regression(X, y)
        xname = list(pd.Index(["bias"]).append(columns))

    # print(results.pvalues)
    # print(results.params)

    print("======")
    print()
    print(dataset)
    print(name)
    print()
    print(results.summary(yname="robustness", xname=xname))

    # print(columns)
    if polynomial is None and len(columns) == 4:
        display_line_on_scatter(X, y, results, columns, name, dataset)

    return results, xname, (np.min(y), np.mean(y), np.median(y), np.max(y))


def get_regression_df(config, df):
    columns = get_columns(config)
    df = df.iloc[0:, [df.columns.get_loc(c) for c in columns]]
    if _activation_function in columns:
        df_with_dummies = pd.get_dummies(df, columns=[_activation_function], prefix="", drop_first=True)
    else:
        df_with_dummies = df
    return df_with_dummies


def get_columns(config):
    if config["use_all_columns"]:
        columns = config["all_columns"]
    else:
        columns = config["small_columns"]
    return columns


def error_plot(dataset, norm, path):
    lower_bounds, lower_bounds_df = get_lower_bounds(norm, path)
    lower_bounds, lower_bounds_df = remove_exapmles_tested_with_only_one_activation_function(lower_bounds, lower_bounds_df)

    upper_bounds, upper_bounds_df = get_merged_upper_bound_df(path + "upper_bound.csv", lower_bounds_df, norm)
    upper_bounds, upper_bounds_df = remove_exapmles_tested_with_only_one_activation_function(upper_bounds, upper_bounds_df)

    HSJA_upper_bound, HSJA_upper_bound_df = get_merged_upper_bound_df(path + "HSJA_upper_bound.csv", lower_bounds_df, norm)
    HSJA_upper_bound, HSJA_upper_bound_df = remove_exapmles_tested_with_only_one_activation_function(HSJA_upper_bound, HSJA_upper_bound_df)

    bim_robustness, bim_robustness_df = get_merged_adversarial_error_df(path + "emprirical_robustness.csv", lower_bounds_df)
    mim_robustness, mim_robustness_df = get_merged_adversarial_error_df(path + "success_rate.csv", lower_bounds_df)

    df = lower_bounds_df
    # all_queries = queries(df, norm) & (df[_filter]%16 != 0)
    all_queries = None  # (df[_filter]%16 != 0)
    # print("len", len(df[all_queries]))

    ys = [_lower_bound]
    # ys = ["accuracy_adversarial"]
    fig = plt.figure(figsize=(4, 12))
    parameters = [_depth, _filter, _activation_function, _kernel]
    for i in range(len(parameters)):
        None
        a = fig.add_subplot(len(parameters), 1, i + 1)
        show_error_bars(df, parameters[i], all_queries, ys, title=dataset + " L_" + norm, fig=a)
    plt.show()

    # error_plot_column(df, _accuracy, query=all_queries)
    # plt.title(dataset + ". Norm: " + norm)
    # plt.show()
    """
    error_plot_column(df, _activation_function, query=all_queries)
    error_plot_column(df, _early_stoppping, query=all_queries)
    error_plot_column(df, _padding_same, query=all_queries)
    error_plot_column(df, _batch_normalization, query=all_queries)"""


def remove_exapmles_tested_with_only_one_activation_function(lower_bounds, lower_bounds_df):
    keep_columns = lower_bounds_df.duplicated([_depth, _filter, _kernel, _l_norm], keep=False)
    lower_bounds_df = lower_bounds_df[keep_columns]
    lower_bounds = lower_bounds[keep_columns]
    return lower_bounds, lower_bounds_df


def main(dataset, norm, columns, fig=None):
    print("======")
    print(dataset, ". Norm: ", norm, sep="")
    print()
    path = (dataset + "/")

    results = None
    results = regression(columns, path, norm, dataset, fig)

    # print(results.pvalues)
    # print(results.params)

    error_plot(dataset, norm, path)

    return results


def regression(columns, path, norm, dataset, fig):
    lower_bounds, lower_bounds_df = get_lower_bounds(norm, path)

    #upper_bounds, upper_bounds_df = get_merged_upper_bound_df(path + "upper_bound.csv", lower_bounds_df, norm)
    #HSJA_upper_bound, HSJA_upper_bound_df = get_merged_upper_bound_df(path + "HSJA_upper_bound.csv", lower_bounds_df, norm)

    #bim_robustness, bim_robustness_df = get_merged_adversarial_error_df(path + "emprirical_robustness.csv", lower_bounds_df)
    #mim_robustness, mim_robustness_df = get_merged_adversarial_error_df(path + "success_rate.csv", lower_bounds_df)

    #adv_robustness, adv_robustness_df = get_merged_adversarial_train_df(path, lower_bounds_df, norm)
    # a = fig.add_subplot(4, 3, datasets.index(dataset)+1)

    lb_results = do_linear_regression(lower_bounds_df, lower_bounds, columns, "Lower bounds", dataset)
    # a = fig.add_subplot(4, 3, datasets.index(dataset)+1+6)
    # ub_results = do_linear_regression(upper_bounds_df, upper_bounds, columns, "Upper bounds", dataset)
    ##do_linear_regression(upper_bounds_df, upper_bounds, columns, "upper_bounds", dataset, polynomial=2)
    #hsja_results = do_linear_regression(HSJA_upper_bound_df, HSJA_upper_bound, columns, "HSJA upper bound", dataset)

    #adv_results = do_linear_regression(adv_robustness_df, adv_robustness, columns, "Lower bounds", dataset)

    adversarial_error_columns = list(columns)
    if _accuracy in adversarial_error_columns:
        adversarial_error_columns.remove(_accuracy)

    # bim_results = do_linear_regression(bim_robustness_df, bim_robustness, adversarial_error_columns, "bim_robustness", dataset)
    # mim_results = do_linear_regression(mim_robustness_df, mim_robustness, adversarial_error_columns, "mim_robustness", dataset)

    return lb_results


def get_lower_bounds(norm, path):
    lower_bounds_df = get_result_df(path)
    lower_bounds_df = lower_bounds_df[queries(lower_bounds_df, norm)]
    lower_bounds = lower_bounds_df["lower_bound"]
    return lower_bounds, lower_bounds_df


def print_result_table(results, norms):
    out_string = ""
    out_string3 = ""
    out_string4 = ""
    num_parameters = len(results[0][0].params)
    for j in range(num_parameters):
        parameter = results[0][1][j].replace("_", "")
        parameter = parameter.capitalize()
        out_string += parameter
        out_string3 += parameter
        out_string4 += "        " + parameter
        for i in range(len(results)):
            coef = results[i][0].params[j]
            pvalue = results[i][0].pvalues[j]
            out_string += " & " + format(coef, '.5f') + " & " + format(pvalue, '.4f')
            out_string3 += " & " + format(coef / results[i][2][1] * 100, '.3f') + "\\%"
            out_string4 += " & " + format(coef / results[i][2][1] * 100, '.3f') + "\\%" + " & " + format(pvalue, '.4f')
        out_string += " \\\\ \hline\n"
        out_string3 += " \\\\ \hline\n"
        out_string4 += " \\\\ \hline\n"
    print(out_string)
    print("")
    print("mean:")
    print(out_string3)
    print("mean and p-values:")
    print(out_string4)

    print()

    l = ["Min", "Mean", "Median", "Max"]
    out_string2 = ""
    for norm in norms:
        out_string2 += f" & {norm}"
    out_string2 += " \\\\ \hline\n"
    for i in range(len(l)):
        out_string2 += f"{l[i]} "
        for j in range(len(results)):
            out_string2 += f" & {format(results[j][2][i], '.4f')}"
        out_string2 += " \\\\ \hline\n"
    print(out_string2)
    print()

    params = ""
    p_values = ""
    for j in range(num_parameters):
        coef = results[0][0].params[j]
        pvalue = results[0][0].pvalues[j]
        params += " & " + format(coef, '.4f')
        p_values += " & " + format(pvalue, '.4f')
    params += " \\\\ \hline\n"
    p_values += " \\\\ \hline\n"
    print(params + p_values)


def queries(df, norm="i"):
    depth_query = (df[_depth] == 3)
    norm_query = (df[_l_norm] == norm)
    depth_query = (df[_depth] == 3)
    kernel_query = (df[_kernel] == 5)
    # filter_query = (df[_filter] < 0.95)
    accuracy_query = (df[_accuracy] >= 0.95)
    padding_query = (~df["padding_same"])
    ub_not_zero = (df[_upper_bound] != 0)
    es_query = (~df["early_stoppping"])
    bn_query = (~df["has_batch_normalization"])
    ac_query = (df[_activation_function] != "arctan") & (df[_activation_function] != "tanh")
    core_query = (~df["Cnn-cert-core"])
    lower_bound_query = (df["lower_bound"] >= 0.03)
    return ub_not_zero & norm_query & accuracy_query  # & ac_query


datasets = ["mnist", "sign-language", "caltechSilhouettes", "rockpaperscissors", "cifar", "GTSRB"]


def main_wrapper():
    with open("config.json") as json_file:
        config = json.load(json_file)
    set_path(config["path"])
    norms = ["i", "2", "1"]
    datasets = ["rockpaperscissors"]

    if config["use_all_columns"]:
        columns = config["all_columns"]
    else:
        columns = config["small_columns"]
    results = []
    for norm in norms:
        # if not config["use_all_columns"]:
        # fig = plt.figure(figsize=(12, 12))
        for dataset in datasets:
            results.append(main(dataset, norm, columns))
        # if not config["use_all_columns"]:
        # plt.show()

    print_result_table(results, norms)


if __name__ == "__main__":
    main_wrapper()

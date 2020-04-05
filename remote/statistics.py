import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
import pydot
import statsmodels.api as sm
import json

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
_upper_bound, _lower_bound = "upper_bound", "lower_bound"


def get_df(columns, path):
    results = pd.read_csv(path + 'results/results.csv')
    accuracy = pd.read_csv(path + 'models_meta.csv')
    """
    results = results[~results["Cnn-cert-core"]]
    results = results[~results["has_batch_normalization"]]
    ac_query = (results[_activation_function] == "ada")
    """
    accuracy_query = (results[_accuracy] >= 0.80)
    bn_query = (~results["has_batch_normalization"])
    results = results[bn_query & accuracy_query]

    results = results[~results[[_file_name, "Cnn-cert-core"]].duplicated(keep="first")]
    #  Merge with
    data_df: pd.DataFrame = pd.merge(left=results, right=accuracy, left_on='file_name', right_on='file_name',
                                     suffixes=("_x", ""))

    df: pd.DataFrame = data_df.drop(["upper_bound"], axis=1)

    ub_df: pd.DataFrame = get_upper_bound_df(path)
    data_df = pd.merge(left=df, right=ub_df, left_on='file_name', right_on='file_name',
                  suffixes=("_x", ""))

    df = data_df.iloc[0:, [data_df.columns.get_loc(c) for c in columns]]
    if _activation_function in columns:
        df_with_dummies = pd.get_dummies(df, columns=[_activation_function], prefix="", drop_first=True)

    return df_with_dummies, data_df["lower_bound"], data_df["accuracy"], data_df["time_per_epoch"], data_df["best_epoch"], data_df[_activation_function], data_df["upper_bound"]


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
    est = sm.OLS(y, X2)
    est2 = est.fit()
    return est2

    """
    model = LinearRegression()
    model.fit(X, y)
    return model"""


def get_y(df):
    return df["lower_bound"]


def plot_single_variable(X, y, name):
    plt.scatter(X, y)

    plt.xlabel(name)
    plt.ylabel("Robustness")
    plt.show()


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
    plt.show()


def error_plot_column(df, column_name, query=None, y="lower_bound"):
    plt.xlabel(column_name)
    plt.ylabel("Robustness")
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
    results = pd.read_csv(path+'results/results.csv')
    return results

def get_upper_bound_df(path):
    results = pd.read_csv(path+'upper_bound.csv')
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

def show_error_bars(df, x, query, ys, title):
    for i in ys:
        error_plot_column(df, x, query, y=i)
    plt.legend()
    plt.title(title)
    plt.show()

def error_plot(dataset, path):
    df: pd.DataFrame = get_result_df(path).drop(["upper_bound"], axis=1)
    ub_df: pd.DataFrame = get_upper_bound_df(path)
    df = pd.merge(left=df, right=ub_df, left_on='file_name', right_on='file_name',
             suffixes=("_x", ""))
    df['acc_rob_low'] = df[_accuracy]*df[_lower_bound]
    df['acc_rob_hi'] = df[_accuracy]*df[_upper_bound]


    depth_query = (df[_depth] == 3)
    kernel_query = (df[_kernel] == 5)
    filter_query = (df[_filter] <= 90)
    accuracy_query = (df[_accuracy] >= 0.80)
    padding_query = (~df["padding_same"])
    es_query = (~df["early_stoppping"])
    bn_query = (~df["has_batch_normalization"])
    ac_query = (df[_activation_function] == "ada")
    core_query = (~df["Cnn-cert-core"])

    #all_queries = None
    all_queries = bn_query & accuracy_query

    ys=[_upper_bound, _lower_bound]
    show_error_bars(df, _filter, all_queries, ys, dataset)
    error_plot_column(df, _filter, query=all_queries, y=ys[1])
    plt.title(dataset)
    plt.show()
    #error_plot_column(df, _depth, query=all_queries)
    #error_plot_column(df, _kernel, query=all_queries)
    """
    error_plot_column(df, _activation_function, query=all_queries)
    error_plot_column(df, _early_stoppping, query=all_queries)
    error_plot_column(df, _padding_same, query=all_queries)
    error_plot_column(df, _batch_normalization, query=all_queries)"""

def set_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)

def main(dataset, config):
    if dataset != "mnist":
        path = (dataset+"/")
    else:
        path = ""

    if config["use_all_columns"]:
        columns = config["all_columns"]
    else:
        columns = config["small_columns"]
    x_df, y_df_lower_bound, y_accuracy, y_time_per_epoch, y_best_epoch, af, upper_bound = get_df(columns, path)
    y = y_df_lower_bound.to_numpy()
    upper_bound_array = upper_bound.to_numpy()

    X, columns, input_dict = do_linear_regression(config, x_df, y, dataset, upper_bound_array)
    """
    y_acc_metric = normalize_0_to_1(y_accuracy.to_numpy())
    do_linear_regression(config, x_df, y_acc_metric, "accuracy")

    y_time_metric = normalize_0_to_1(y_time_per_epoch.to_numpy() * y_best_epoch.to_numpy())
    do_linear_regression(config, x_df, normalize_0_to_1(y_time_per_epoch.to_numpy() * y_best_epoch.to_numpy()), "speed")

    metric = (normalize_0_to_1(y) * y_acc_metric) / (y_time_metric+1.e-17)
    do_linear_regression(config, x_df, metric, "metric")"""

    for i in columns:
        if config["print_scatter"]:
            plot_single_variable(input_dict[i], y, i)
    if config["use_error_plot"]:
        error_plot(dataset, path)
    # plot_3d(input_dict[_filter], input_dict[_depth], y, af)
    if config["use_decicion_tree"]:
        None
        #decicion_tree(X, y, columns=list(columns.to_numpy()))


def normalize_0_to_1(x):
    return (x - min(x)) / (max(x) - min(x))


def do_linear_regression(config, x_df, y, name, upper_bound_array):
    print("====================")
    print()
    print(name)
    columns = x_df.keys()
    X, input_dict = get_numppy_arrays(x_df)

    linear_regression_model = linear_regression(X, y)
    poly_features, plynomial_regression_model = poly_reg(X, y, config["p"])
    #  poly_features3, plynomial_regression_model3 = poly_reg(X, y, 3)

    ub_linear_regression_model = linear_regression(X, upper_bound_array)
    ub_poly_features, ub_plynomial_regression_model = poly_reg(X, upper_bound_array, config["p"])
    #  ub_poly_features3, ub_plynomial_regression_model3 = poly_reg(X, upper_bound_array, 3)

    print(linear_regression_model.summary(yname="robustness", xname=list(pd.Index(["bias"]).append(columns))))
    #print(plynomial_regression_model.summary(yname="robustness", xname=poly_features.get_feature_names(x_df.columns)))

    print("========")
    print()
    print("upper")

    #print(ub_linear_regression_model.summary(yname="robustness", xname=list(pd.Index(["bias"]).append(columns))))
    #print(ub_plynomial_regression_model.summary(yname="robustness", xname=poly_features.get_feature_names(x_df.columns)))

    return X, columns, input_dict



with open("config.json") as json_file:
    config = json.load(json_file)
set_path(config["path"])
datasets = ["mnist", "cifar", "GTSRB", "rockpaperscissors", "sign-language", "caltechSihouettes"]
for i in datasets:
    main(i, config)

"""
model = LinearRegression()
model.fit(data, y)

print(len(x_df.columns))
print(len(model.coef_))
print()
for i in range(len(x_df.columns)):
    print("{}: {}".format(x_df.columns[i], model.coef_[i]))

print()

def plot_linear_variable(X, y, name):
    plt.scatter(X, y)
    plt.show()


plot_linear_variable(filter_data, y, _filter)
plot_linear_variable(kernel_data, y, _kernel)

model2 = LinearRegression()
poly2 = PolynomialFeatures(degree=2)
x = poly2.fit_transform(data, y)
model2.fit(x, y)

print(len(poly2.get_feature_names(x_df.columns)))
print(len(model2.coef_))
for i in range(len(model2.coef_)):
    print("{}: {}".format(poly2.get_feature_names(x_df.columns)[i], model2.coef_[i]))

def plot_variable(X, y=y, poly_features=poly2):
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    X_plot_poly = poly_features.fit_transform(X_plot)
    plt.plot(X, y, "b.")
    plt.plot(X_plot_poly, model.predict(X_plot_poly), '-r')
    plt.show()"""

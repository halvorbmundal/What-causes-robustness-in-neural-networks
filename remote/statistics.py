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


def get_df(columns):
    results = pd.read_csv('results/results.csv')
    accuracy = pd.read_csv('models_meta.csv')
    results = results[~results["Cnn-cert-core"]]
    padding_query = (~results["padding_same"])
    es_query = (results["early_stoppping"])
    ac_query = (results[_activation_function] != "arctan")
    results = results[padding_query & es_query & ac_query]
    results = results[~results[[_file_name, "Cnn-cert-core"]].duplicated(keep="first")]
    data_df: pd.DataFrame = pd.merge(left=results, right=accuracy, left_on='file_name', right_on='file_name',
                                     suffixes=("_x", ""))
    df_with_dummies = data_df.iloc[0:, [data_df.columns.get_loc(c) for c in columns]]
    if _activation_function in columns:
        df_with_dummies = pd.get_dummies(df_with_dummies,
                                     columns=[_activation_function], prefix="", drop_first=True)
    return df_with_dummies, data_df["lower_bound"], data_df[_activation_function]


def get_numppy_arrays(df):
    x = df.to_numpy()
    x_individual = {}
    for i in df.keys():
        x_individual[i] = df[i].to_numpy()
    return x, x_individual


def linear_regression(X, y):
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


def error_plot_column(df, column_name, query=None):
    plt.xlabel(column_name)
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

        avg_lower.append(dff["lower_bound"].mean())
        std_lower.append(dff["lower_bound"].std())

    plt.errorbar(x, avg_lower, std_lower)
    plt.show()


# def err_plot_query():
#    (data["depth"] == 6)

def get_result_df():
    results = pd.read_csv('results/results.csv')

    padding_query = (~results["padding_same"])
    es_query = (results["early_stoppping"])
    ac_query = (results[_activation_function] != "arctan")
    results = results[padding_query & es_query & ac_query]

    df = results[~results[[_file_name, "Cnn-cert-core"]].duplicated(keep="first")]
    df = df[~df["Cnn-cert-core"]]
    return df


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


def error_plot():
    df: pd.DataFrame = get_result_df()
    depth_query = (df[_depth] == 3)
    kernel_query = (df[_kernel] == 5)
    filter_query = (df[_filter] <= 90)
    padding_query = (~df["padding_same"])
    es_query = (df["early_stoppping"])
    ac_query = (df[_activation_function] == "ada")
    all_queries = None
    #all_queries = ac_query & depth_query & filter_query & kernel_query

    error_plot_column(df, _kernel, query=all_queries)
    error_plot_column(df, _depth, query=all_queries)
    error_plot_column(df, _filter, query=all_queries)
    error_plot_column(df, _activation_function, query=all_queries)
    error_plot_column(df, _early_stoppping, query=all_queries)
    error_plot_column(df, _padding_same, query=all_queries)

def set_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)

def main():
    with open("config.json") as json_file:
        config = json.load(json_file)
    set_path(config["path"])

    if config["use_all_columns"]:
        columns = config["all_columns"]
    else:
        columns = config["small_columns"]

    x_df, y_df, af = get_df(columns)
    y = y_df.to_numpy()
    columns = x_df.keys()
    X, input_dict = get_numppy_arrays(x_df)

    linear_regression_model = linear_regression(X, y)
    poly_features, plynomial_regression_model = poly_reg(X, y, config["p"])
    poly_features3, plynomial_regression_model3 = poly_reg(X, y, 3)
    print(linear_regression_model.summary(yname="robustness", xname=list(pd.Index(["bias"]).append(columns))))
    print(plynomial_regression_model.summary(yname="robustness", xname=poly_features.get_feature_names(x_df.columns)))
    print(plynomial_regression_model3.summary(yname="robustness", xname=poly_features3.get_feature_names(x_df.columns)))

    for i in columns:
        if config["print_scatter"]:
            plot_single_variable(input_dict[i], y, i)
    if config["use_error_plot"]:
        error_plot()
    # plot_3d(input_dict[_filter], input_dict[_depth], y, af)
    if config["use_decicion_tree"]:
        decicion_tree(X, y, columns=list(columns.to_numpy()))


main()

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

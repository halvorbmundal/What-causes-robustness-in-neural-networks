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

from sklearn.datasets import load_iris
from sklearn import tree

_dataset = "dataset"
_file_name = "file_name"
_kernel = "kernel"
_depth = "depth"
_accuracy = "accuracy"
_filter = "filter"
_activation_function = "activation_function"


def get_df(columns):
    results = pd.read_csv('results/results.csv')
    accuracy = pd.read_csv('models_meta.csv')
    results = results[~results[_file_name].duplicated(keep="first")]
    data_df: pd.DataFrame = pd.merge(left=results, right=accuracy, left_on='file_name', right_on='file_name',
                                     suffixes=("_x", ""))
    df_with_dummies = pd.get_dummies(data_df.iloc[0:, [data_df.columns.get_loc(c) for c in columns]],
                          columns=[_activation_function])
    return df_with_dummies, data_df["lower_bound"], data_df[_activation_function]



def get_numppy_arrays(df):
    x = df.to_numpy()
    x_individual = {}
    for i in df.keys():
        x_individual[i] = df[i].to_numpy()
    return x, x_individual


def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


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
    finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

def decicion_tree(X, y):
    clf = tree.DecisionTreeRegressor(random_state=0, max_depth=3)
    clf = clf.fit(X, y)
    tree.plot_tree(clf)
    #tree.export_graphviz(clf)
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


#def err_plot_query():
#    (data["depth"] == 6)

def get_result_df():
    results = pd.read_csv('results/results.csv')
    df = results[~results[_file_name].duplicated(keep="first")]
    return df

def error_plot():
    df = get_result_df()
    depth_query = (df[_depth] == 3)
    kernel_query = (df[_kernel] == 5)
    filter_query = (df[_filter] <= 90)
    ac_query = (df[_activation_function] == "ada")
    all_queries = depth_query & kernel_query & filter_query & ac_query

    error_plot_column(df, _kernel, query=all_queries)
    error_plot_column(df, _depth, query=all_queries)
    error_plot_column(df, _filter, query=all_queries)

def main():
    columns = [_kernel, _depth, _filter, _activation_function]
    x_df, y_df, af = get_df(columns)
    y = y_df.to_numpy()
    columns = x_df.keys()
    X, input_dict = get_numppy_arrays(x_df)
    linear_regression_model = linear_regression(X, y)
    for i in columns:
        if False:
            plot_single_variable(input_dict[i], y, i)
    for i in range(len(x_df.columns)):
        print("{}: {}".format(x_df.columns[i], linear_regression_model.coef_[i]))

    error_plot()
    #plot_3d(input_dict[_filter], input_dict[_depth], y, af)
    #decicion_tree(X, y)


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

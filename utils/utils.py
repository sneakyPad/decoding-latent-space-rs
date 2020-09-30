import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import surprise
from sklearn import manifold, decomposition
from sklearn.metrics import mean_squared_error

from torchsummaryX import summary

def calculate_metrics(y_actual, y_predicted):
    #RMSE
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)
    print("RMSE :{}".format(rmse))

    #MSE
    mse = mean_squared_error(y_actual, y_predicted, squared=True)
    print("MSE :{}".format(mse))
    return rmse,mse

def print_nn_summary(model):
    example_input = torch.zeros((1, 9724))
    summary(model, example_input)

def plot_mce(model, neptune_logger, max_epochs):
    avg_mce = model.avg_mce

    ls_x = []
    ls_y = []

    for key, val in avg_mce.items():
        neptune_logger.log_metric('MCE_' + key, val)
        ls_x.append(key)
        ls_y.append(val)
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 12))

    sns_plot = sns.barplot(x=ls_x, y=ls_y)
    fig = sns_plot.get_figure()
    # fig.set_xticklabels(rotation=45)
    plt.xticks(rotation=70)
    plt.tight_layout()
    fig.savefig("./results/images/mce_epochs_" + str(max_epochs) + ".png")

def load_json_as_dict(name):
    with open('../data/generated/' + name, 'r') as file:
        id2names = json.load(file)
        return id2names


def ls_columns_to_dfrows(ls_val, column_base_name):
    print(ls_val.shape)
    ls_columns = [column_base_name + str(i) for i in range(1, ls_val.shape[1] + 1)]
    print(ls_columns)
    df_z = pd.DataFrame(data=ls_val, columns=ls_columns)
    print(df_z.columns)
    df_piv = df_z.melt(var_name='cols', value_name='values')  # Transforms it to: _| cols | vals|
    return df_piv

def plot_catplot(df, title):
    # plt.xticks(rotation=45)
    g=sns.catplot(x="cols", y="values", data=df).set(title=title)
    g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=65)
    plt.show()

def plot_swarmplot(df, title):
    # plt.xticks(rotation=45)
    ax=sns.swarmplot(x="cols", y="values", data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
    ax.set(title=title)
    plt.show()

def plot_violinplot(df, title):
    # plt.xticks(rotation=45)
    ax=sns.violinplot(x="cols", y="values", data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
    ax.set(title=title)
    plt.show()




def plot_distribution(df_melted, title):
    # plt.figure(figsize=(10,10))
    # sns.violinplot(x=foo[:,0])
    sns.displot(df_melted, x="values", hue="cols", kind="kde", rug=True).set(title=title)
    plt.show()

#PCA
def apply_pca(np_x):
    pca = decomposition.PCA(n_components=2)
    # print(np_x)
    pca.fit(np_x)
    X = pca.transform(np_x)
    # print(X)
    return X

def plot_2d_pca(np_x, title):
    df_pca = pd.DataFrame(np_x, columns=['pca_1', 'pca_2'])
    sns.scatterplot(data=df_pca, x="pca_1", y="pca_2").set(title=title)
    # g.set_xticklabels(g.get_xticklabels(), rotation=45)
    # plt.setp(ax.get_xticklabels(), rotation=45)
    # plt.scatter(np_x[:,0], np_x[:,1]) #only on numpy array
    plt.show()




def plot_results(model, neptune_logger, max_epochs):
    sns.set_style("whitegrid")
    sns.set_theme(style="ticks")

    # Apply PCA on Data an plot it afterwards
    np_z_pca = apply_pca(model.np_z)
    plot_2d_pca(np_z_pca, "PCA applied on Latent Factors w/ dim: " + str(model.no_latent_factors))

    # Plot the probability distribution of latent layer
    df_melted = ls_columns_to_dfrows(ls_val=model.np_z, column_base_name="LF: ")
    plot_distribution(df_melted, 'Probability Distribution of Laten Factors (z)')
    plot_catplot(df_melted, "Latent Factors")
    plot_swarmplot(df_melted, "Latent Factors")
    plot_violinplot(df_melted, "Latent Factors")

    plot_mce(model, neptune_logger, max_epochs)

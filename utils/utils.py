import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import manifold, decomposition
from sklearn.metrics import mean_squared_error

from torchsummaryX import summary
from datetime import datetime
import os

def create_experiment_directory():
    # datetime object containing current date and time
    now = datetime.now()

    print("now =", now)
    dt_string = now.strftime("%d-%m-%Y-%H_%M_%S")
    print("date and time =", dt_string)

    # define the name of the directory to be created
    path = "results/images/" + dt_string + "/"

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    return path

def calculate_metrics(y_actual, y_predicted):
    #RMSE
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)
    # print("RMSE :{}".format(rmse))

    #MSE
    mse = mean_squared_error(y_actual, y_predicted, squared=True)
    # print("MSE :{}".format(mse))
    return rmse,mse

def print_nn_summary(model):
    example_input = torch.zeros((1, 9724))
    summary(model, example_input)

def save_figure(fig, experiment_path, name, dct_params):
    image_name = name + "_" + "_".join(str(val)+"_"+key for key, val in dct_params.items())
    fig.savefig(experiment_path + image_name + ".png")


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

def save_dict_as_json(dct, name):
    with open('../data/generated/' + name, 'w') as file:
        json.dump(dct, file, indent=4, sort_keys=True)

def ls_columns_to_dfrows(ls_val, column_base_name):
    print(ls_val.shape)
    ls_columns = [column_base_name + str(i) for i in range(1, ls_val.shape[1] + 1)]
    # print(ls_columns)
    df_z = pd.DataFrame(data=ls_val, columns=ls_columns)
    # print(df_z.columns)
    df_piv = df_z.melt(var_name='cols', value_name='values')  # Transforms it to: _| cols | vals|
    return df_piv

def plot_catplot(df, title, experiment_path, dct_params):
    # plt.xticks(rotation=45)
    g=sns.catplot(x="cols", y="values",s=3, data=df).set(title=title)
    g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=65)
    save_figure(g, experiment_path, 'catplot', dct_params)

    plt.show()

def plot_swarmplot(df, title, experiment_path,dct_params):
    # plt.xticks(rotation=45)
    ax=sns.swarmplot(x="cols", y="values",s=3, data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
    ax.set(title=title)

    save_figure(ax.get_figure(), experiment_path, 'swarmplot', dct_params)


    plt.show()

def plot_violinplot(df, title,experiment_path,dct_params):
    # plt.xticks(rotation=45)
    ax=sns.violinplot(x="cols", y="values", data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
    ax.set(title=title)

    save_figure(ax.get_figure(), experiment_path, 'violinplot', dct_params)
    plt.show()

def plot_distribution(df_melted, title, experiment_path, dct_params):
    # plt.figure(figsize=(10,10))
    # sns.violinplot(x=foo[:,0])
    fig = sns.displot(df_melted, x="values", hue="cols", kind="kde", rug=True).fig
    fig.suptitle(title)
    save_figure(fig, experiment_path, 'distribution', dct_params)

    plt.show()

#PCA
def apply_pca(np_x):
    pca = decomposition.PCA(n_components=2)
    # print(np_x)
    pca.fit(np_x)
    X = pca.transform(np_x)
    # print(X)
    return X

def plot_2d_pca(np_x, title, experiment_path, dct_params):
    df_pca = pd.DataFrame(np_x, columns=['pca_1', 'pca_2'])
    fig = sns.scatterplot(data=df_pca, x="pca_1", y="pca_2").get_figure()
    fig.suptitle(title)

    # g.set_xticklabels(g.get_xticklabels(), rotation=45)
    # plt.setp(ax.get_xticklabels(), rotation=45)
    # plt.scatter(np_x[:,0], np_x[:,1]) #only on numpy array
    save_figure(fig, experiment_path,'pca', dct_params)
    plt.show()


def plot_mce_by_latent_factor(df_mce, title, experiment_path, dct_params):


    plt.figure(figsize=(30, 20))
    df_mce['category'] = df_mce.index
    df_piv = df_mce.melt(id_vars='category', var_name='latent_factor', value_name='mce')
    fig = sns.catplot(x='latent_factor',
                      y='mce',
                      hue='category',
                      data=df_piv,
                      kind='bar',
                      legend_out=False,
                      aspect=1.65
                      )
    plt.title(title, fontsize=17, y=1.08)

    plt.legend(bbox_to_anchor=(0.5, -0.25),  # 1.05, 1
               loc='upper center',  # 'center left'
               borderaxespad=0.,
               fontsize=8,
               ncol=5)

    plt.tight_layout()
    save_figure(fig, experiment_path,'mce_latent_factor', dct_params)
    plt.show()


def plot_parallel_plot(df_mce, title, experiment_path, dct_params):
    df_flipped = df_mce.transpose() # columns and rows need to be flipped
    ls_lf = [i for i in range(0, df_flipped.shape[0])]
    df_flipped['latent_factors'] = ls_lf
    ax = pd.plotting.parallel_coordinates(
        df_flipped, 'latent_factors', colormap='viridis')

    ax.xaxis.set_tick_params(labelsize=6)
    fig = ax.get_figure()
    plt.title(title, fontsize=20, y=1.08)
    # plt.xlabel('xlabel', fontsize=10)
    plt.xticks(rotation=90)

    plt.tight_layout()
    save_figure(fig, experiment_path,'parallel_plot', dct_params)
    plt.show()

def plot_KLD(ls_kld, title, experiment_path, dct_params):
    # ls_kld =[2,200,2000,1800,2000,1500]
    df_kld = pd.DataFrame(data=ls_kld, columns=['KLD'])
    ax = sns.lineplot(data = df_kld, x=df_kld.index, y="KLD")


    plt.show()


def plot_pairplot_lf(model, title, experiment_path, dct_params):
    df_kld_matrix = pd.DataFrame(data=model.kld_matrix,
                                 columns=[str(i) for i in range(0, model.kld_matrix.shape[1])])
    fig = sns.pairplot(df_kld_matrix, corner=True, aspect=1.65).fig
    plt.title(title, fontsize=17, y=1.08)
    plt.tight_layout()
    save_figure(fig, experiment_path, 'lf_correlation', dct_params)
    plt.show()



def plot_results(model, experiment_path, dct_params):

    sns.set_style("whitegrid")
    # sns.set_theme(style="ticks")
    # df_mce_results = pd.read_json('../data/generated/mce_results.json')

    # Apply PCA on Data an plot it afterwards
    np_z_pca = apply_pca(model.np_z_test)
    plot_2d_pca(np_z_pca, "PCA applied on Latent Factors w/ dim: " + str(model.no_latent_factors), experiment_path,dct_params)

    # Plot the probability distribution of latent layer
    df_melted = ls_columns_to_dfrows(ls_val=model.np_z_test, column_base_name="LF: ")
    plot_distribution(df_melted, 'Probability Distribution of Laten Factors (z)', experiment_path,dct_params)
    plot_catplot(df_melted, "Latent Factors", experiment_path,dct_params)
    plot_swarmplot(df_melted, "Latent Factors", experiment_path,dct_params)
    plot_violinplot(df_melted, "Latent Factors", experiment_path,dct_params)
    # plot_mce_by_latent_factor(df_mce_results.copy(), 'MCE sorted by Latent Factor', experiment_path, dct_params) ##make a copy otherwise the original df is altered,
    # plot_parallel_plot(df_mce_results.copy(), 'MCE for different Metadata', experiment_path, dct_params)##make a copy otherwise the original df is altered,
    plot_KLD(model.ls_kld, 'KLD over Epochs (Training)', experiment_path, dct_params)
    plot_pairplot_lf(model, 'Correlation of Latent Factors', experiment_path, dct_params)

    # plot_mce(model, neptune_logger, max_epochs) #TODO Change method to process multiple entries
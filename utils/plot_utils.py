import torch
import json
import seaborn as sns
import pandas as pd
from sklearn import decomposition
import math
from torchsummaryX import summary
from datetime import datetime
import random
from tqdm import tqdm
import ast
import itertools
from collections import Counter
import plotly.express as px
import matplotlib.pyplot as plt
import os
import numpy as np


def create_heatmap(np_arr, title, y_label, x_label, experiment_path, dct_params):
    ax = sns.heatmap(np_arr)
    # plt.suptitle(title)
    ax.set(title=title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    save_figure(ax.get_figure(), experiment_path, 'heatmap-user-item', dct_params)
    plt.show()

def print_nn_summary(model, size):
    example_input = torch.zeros((1, size))
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
    if(np_x.shape[1] > 1):
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

def plot_mce_wo_kld(df_mce, title, experiment_path, dct_params):
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
    save_figure(fig, experiment_path,title, dct_params)
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
    save_figure(fig, experiment_path,'mce-latent-factor', dct_params)
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
    save_figure(fig, experiment_path,'parallel-plot', dct_params)
    plt.show()

def plot_KLD(ls_kld, title, experiment_path, dct_params):
    # ls_kld =[2,200,2000,1800,2000,1500]
    df_kld = pd.DataFrame(data=ls_kld, columns=['KLD'])
    ax = sns.lineplot(data = df_kld, x=df_kld.index, y="KLD")


    plt.show()


def plot_pairplot_lf_kld(model, title, experiment_path, dct_params):
    df_kld_matrix = pd.DataFrame(data=model.kld_matrix,
                                 columns=[str(i) for i in range(0, model.kld_matrix.shape[1])])
    fig = sns.pairplot(df_kld_matrix, corner=True, aspect=1.65).fig
    # for ax in fig.axes:
    #     ax.set_xlim(-3, 3)
    #     ax.set_ylim(-3, 3)


    fig.suptitle(title)
    # plt.title(title, fontsize=17, y=1.08)
    plt.tight_layout()
    plt.ylabel('Latent Factor')
    plt.xlabel('Latent Factor')
    save_figure(fig, experiment_path, 'lf-correlation-kld', dct_params)
    plt.show()

def plot_kld_of_latent_factor(model, title, experiment_path, dct_params):
    df_kld_matrix = pd.DataFrame(data=model.kld_matrix,
                                 columns=[str(i) for i in range(0, model.kld_matrix.shape[1])])

    df = df_kld_matrix.agg("mean")  # .reset_index()


    ax = df.plot.bar(stacked=False, rot=0)
    plt.title(title, fontsize=17, y=1.08)
    plt.ylabel('KLD')
    plt.xlabel('Latent Factor')
    fig = ax.get_figure()
    save_figure(fig, experiment_path, 'kld-lf-mean', dct_params)

    plt.show()

def plot_pairplot_lf_z(model, title, experiment_path, dct_params):
    df_z_matrix = pd.DataFrame(data=model.np_z_test,
                                 columns=[str(i) for i in range(0, model.np_z_test.shape[1])])
    df_z_matrix['y'] = model.test_y
    fig = sns.pairplot(df_z_matrix, corner=True, aspect=1.65, hue='y').fig

    fig.suptitle(title)
    # plt.title(title, fontsize=17, y=1.08)
    plt.tight_layout()
    save_figure(fig, experiment_path, 'lf-correlation-z', dct_params)
    plt.show()


def plot_3D_lf_z(model, title, experiment_path, dct_params):
    # df = px.data.iris()
    df_z_matrix = pd.DataFrame(data=model.np_z_test,
                               columns=[str(i) for i in range(0, model.np_z_test.shape[1])])
    df_z_matrix['y'] = model.test_y
    if(df_z_matrix.shape[1]==3):
        fig = px.scatter_3d(df_z_matrix, x='0', y='1', color='y', opacity=0.7)
    else:
        fig = px.scatter_3d(df_z_matrix, x='0', y='1', z='2', color='y', opacity=0.7)

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    # plt.tight_layout()
    fig.show()


def polar_plot(model, title, experiment_path, dct_params):

    df_z_matrix = pd.DataFrame(data=model.np_z_test,
                               columns=[str(i) for i in range(0, model.np_z_test.shape[1])])
    # if(df_z_matrix.shape[1]!=2):
    #     df_z_matrix = df_z_matrix.iloc[:,1:3]
        # return

    df_z_matrix['y'] = model.test_y

    # xs = np.arange(7)
    # ys = xs ** 2

    fig = plt.figure(figsize=(5, 10))
    ax = plt.subplot(2, 1, 1)

    # If we want the same offset for each text instance,
    # we only need to make one transform.  To get the
    # transform argument to offset_copy, we need to make the axes
    # first; the subplot command above is one way to do this.
    # trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
    #                                        x=0.05, y=0.10, units='inches')

    ls_zero = df_z_matrix['0']
    ls_one = df_z_matrix['1']


    for x, y, factor in zip(ls_zero, ls_one, df_z_matrix['y']):
        if(factor =='genres'):
            plt.plot(x, y, 'ro')

        else:
            plt.plot(x, y, 'bo')

        # plt.text(x, y, '%d, %d' % (int(x), int(y)))

    # offset_copy works for polar plots also.
    ax = plt.subplot(2, 1, 2, projection= None) #projection= polar

    # trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
    #                                        y=6, units='dots')

    for xs, ys, factor in zip(ls_zero, ls_one, df_z_matrix['y']):
        r = np.sqrt(xs ** 2 + ys ** 2)
        t = np.arctan2(ys, xs)

        # plt.polar(t, r, 'ro')
        if (factor == 'genres'):
            plt.plot(t, r, 'ro')

        else:
            plt.plot(t, r, 'bo')
        # plt.text(xs, ys, '%d, %d' % (int(x), int(y)), horizontalalignment='center', verticalalignment='bottom')

    save_figure(fig, experiment_path, 'polar', dct_params)
    plt.show()

def plot_gen_factors2latent_factors(model, title, experiment_path, dct_params):
    df_z_matrix = pd.DataFrame(data=model.np_z_test,
                                 columns=[str(i) for i in range(0, model.np_z_test.shape[1])])
    df_z_matrix['y'] = model.test_y
    df = df_z_matrix.groupby('y').agg("mean")#.reset_index()

    #Create for mean
    ax = df.plot.bar(stacked=False,rot=0)
    plt.title('gen-factors2latent-factors-mean', fontsize=17, y=1.08)
    plt.tight_layout()
    fig = ax.get_figure()
    plt.ylabel('z value')
    # fig.suptitle(title)
    save_figure(fig, experiment_path, 'gen-factors2latent-factors-mean', dct_params)
    plt.show()

    #Do the same vor sum
    df_two = df_z_matrix.groupby('y').agg("sum")#.reset_index()
    ax = df_two.plot.bar(stacked=False,rot=0)
    fig = ax.get_figure()
    # fig.suptitle(title)
    plt.title('gen-factors2latent-factors-sum', fontsize=17, y=1.08)
    plt.ylabel('z value')
    save_figure(fig, experiment_path, 'gen-factors2latent-factors-sum', dct_params)
    plt.show()



def plot_results(model, experiment_path_test, experiment_path_train, dct_params):

    sns.set_style("whitegrid")
    # sns.set_theme(style="ticks")
    df_mce_results = pd.read_json(experiment_path_test+'/mce_results.json')#../data/generated/mce_results.json'
    df_mce_wo_kld_results = pd.read_json(experiment_path_train+'/mce_results_wo_kld.json')#../data/generated/mce_results.json'
    # df_mce_wo_kld_results2 = pd.read_json(experiment_path+'/mce_results_wo_kld2.json')#../data/generated/mce_results.json'
    exp_path_img = experiment_path_test + "images/"

    # Apply PCA on Data an plot it afterwards
    np_z_pca = apply_pca(model.np_z_test)
    plot_2d_pca(np_z_pca, "PCA applied on Latent Factors w/ dim: " + str(model.no_latent_factors), exp_path_img,dct_params)

    plot_gen_factors2latent_factors(model, 'gen2latent', exp_path_img, dct_params)
    polar_plot(model, 'Correlation of Latent Factors for Z', exp_path_img, dct_params)

    # Plot the probability distribution of latent layer
    df_melted = ls_columns_to_dfrows(ls_val=model.np_z_test, column_base_name="LF: ")
    plot_distribution(df_melted, 'Probability Distribution of Latent Factors (z)', exp_path_img,dct_params)
    plot_catplot(df_melted, "Latent Factors", exp_path_img,dct_params)
    plot_swarmplot(df_melted, "Latent Factors", exp_path_img,dct_params)
    plot_violinplot(df_melted, "Latent Factors", exp_path_img,dct_params)
    # plot_mce_by_latent_factor(df_mce_results.copy(), 'MCE sorted by Latent Factor', exp_path_img, dct_params) ##make a copy otherwise the original df is altered,
    # plot_mce_wo_kld(df_mce_wo_kld_results.copy(), 'MCE sorted by Latent Factor - wo KLD', exp_path_img, dct_params) ##make a copy otherwise the original df is altered,
    # plot_mce_wo_kld(df_mce_wo_kld_results2.copy(), 'MCE sorted by Latent Factor - wo KLD 2', exp_path_img, dct_params) ##make a copy otherwise the original df is altered,
    # plot_parallel_plot(df_mce_results.copy(), 'MCE for different Metadata', exp_path_img, dct_params)##make a copy otherwise the original df is altered,
    plot_KLD(model.ls_kld, 'KLD over Epochs (Training)', exp_path_img, dct_params)
    plot_pairplot_lf_z(model, 'Correlation of Latent Factors for Z', exp_path_img, dct_params)
    plot_pairplot_lf_kld(model, 'Correlation of Latent Factors for KLD', exp_path_img, dct_params)
    plot_kld_of_latent_factor(model, 'Mean of KLD by Latent Factor', exp_path_img, dct_params)

    # plot_3D_lf_z(model, '3D of Z-Values for LF', exp_path_img, dct_params)

    # plot_mce(model, neptune_logger, max_epochs) #TODO Change method to process multiple entries
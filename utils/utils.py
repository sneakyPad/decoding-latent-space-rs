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
import random
import numpy as np
from tqdm import tqdm
import ast
import itertools
from collections import Counter
from time import sleep
def my_eval(expression):
    try:
        return ast.literal_eval(str(expression))
    except SyntaxError: #e.g. a ":" or "(", which is interpreted by eval as command
            return expression
    except ValueError: #e.g. an entry is nan, in that case just return an empty string
        return ''


def compute_relative_frequency(df_meta):
    print('Compute relative frequency for all columns and their attributes...')
    #Goal is:
    #Cast:
        # Tom Hanks: 0,3%
        # Matt Damon: 0,2%
    # fpp =np.vstack(df_meta['genres'].values)
    # np_array = df_meta['genres'].values
    # tmp_list = []
    # for element in np_array:
    #     tmp_list.extend(eval(element))

    # print(fpp)len_crawled_ids
    #TODO Implement my eval: https://stackoverflow.com/questions/31423864/check-if-string-can-be-evaluated-with-eval-in-python
    dct_rel_freq={}
    for column in tqdm(df_meta.columns, total=len(df_meta.columns)):
        print('Column: {}'.format(column))
    #     ls_ls_casted=[]
    #     for str_elem in df_meta[column].values:
    #         str_elem= str(str_elem)#.replace(':','').replace('(','').replace(')','')
    #         try:
    #             ls_ls_casted.append(ast.literal_eval(str_elem))
    #         except SyntaxError:
    #             ls_ls_casted.append(str_elem)

        ls_ls_casted = [my_eval(str_elem) for str_elem in df_meta[column].values] #cast encoded lists to real list
        # ls_ls_casted = [json.loads(str(str_elem)) for str_elem in df_meta[column].values] #cast encoded lists to real list
        try:
            if(type(ls_ls_casted[0]) == list):
                merged_res = itertools.chain(*ls_ls_casted) #join all lists to one single list
                ls_merged = list(merged_res)
            else:
                ls_merged = ls_ls_casted
            if(column not in ['Unnamed: 0', 'unnamed_0']):
                c = Counter(ls_merged)
                dct_counter = {str(key): value for key, value in c.items()}
                dct_rel_freq[column]={}
                dct_rel_freq[column]['absolute'] = dct_counter
                # print('Column: {}\n\t absolute:{}'.format(dct_rel_freq[column]['absolute']))

                dct_rel_attribute = {str(key): value / sum(c.values()) for key, value in dct_counter.items()} #TODO create a dict with key val
                dct_rel_freq[column]['relative'] = dct_rel_attribute
                # print('\t relative:{}'.format(dct_rel_freq[column]['relative']))

        except TypeError:
            print('TypeError for Column:{} and ls_ls_casted:{} and *ls_ls_casted:{}'.format(column, ls_ls_casted, *ls_ls_casted))


    return dct_rel_freq
    # save_dict_as_json(dct_rel_freq, 'relative_frequency.json')




def create_synthetic_data():
    no_samples =50
    genres = ['Crime', 'Mystery', 'Thriller', 'Action', 'Drama', 'Romance','Comedy', 'War','Adventure', 'Family']
    year = ['1980', '1990', '2000', '2010', '2020']
    stars = ['Tom Hanks', 'Tim Allen', 'Don Rickles','Robin Williams', 'Kirsten Dunst', 'Bonnie Hunt']
    rating = ['7', '8', '9', '10']

    dct_base_data ={'genres': genres, 'year': year, 'stars': stars, 'rating': rating}
    ls_movies = []

    #genre-users
    ls_attributes = ['genres', 'year', 'stars', 'rating']
    n_users = 600
    n_movies = len(ls_attributes * no_samples)
    np_user_item = np.zeros((n_users,n_movies),dtype="float32")


    for attribute in ls_attributes:
        for i in range(no_samples):
            movie = {}
            movie[attribute] = [dct_base_data[attribute][0]]

            for other_attribute in ls_attributes:
                if(other_attribute == attribute):
                    continue
                if(other_attribute == 'rating' or other_attribute == 'year'):
                    movie[other_attribute] = random.choices(dct_base_data[other_attribute], k=1)
                else:
                    movie[other_attribute] = random.choices(dct_base_data[other_attribute], k=2)


            ls_movies.append(movie)


    df_synthentic_data = pd.DataFrame(columns=['genres', 'year', 'stars', 'rating'], data=ls_movies)
    df_synthentic_data['id'] = df_synthentic_data.index
    df_synthentic_data.to_csv('../data/generated/syn.csv', index=False)

    no_users_attribute_specific = int(n_users / len(ls_attributes))
    for i in range(0, len(ls_attributes)):
        end = (i+1) * no_samples
        start = end - no_samples
        sr_ids = df_synthentic_data.loc[start:end]['id']

        for idx in range(no_users_attribute_specific):
            no_of_seen_items = int(random.uniform(20, 40))
            seen = random.sample(list(sr_ids.values), k=no_of_seen_items)
            user_idx = i * no_users_attribute_specific + idx
            np_user_item[user_idx,seen] = 1

    # print(ls_movies)

    return np_user_item


def create_experiment_directory():
    # datetime object containing current date and time
    now = datetime.now()

    print("now =", now)
    dt_string = now.strftime("%d-%m-%Y-%H_%M_%S")
    print("date and time =", dt_string)

    # define the name of the directory to be created
    path = "results/generated/" + dt_string + "/"

    try:
        os.mkdir(path)
        os.mkdir(path+"images/")

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


def load_json_as_dict(name):
    with open('../data/generated/' + name, 'r') as file:
        id2names = json.load(file)
        return id2names

def save_dict_as_json(dct, name, path=None):
    if(path):
        path = path
    else:
        path = '../data/generated/'

    with open(path + name, 'w') as file:
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

    # Plot the probability distribution of latent layer
    df_melted = ls_columns_to_dfrows(ls_val=model.np_z_test, column_base_name="LF: ")
    plot_distribution(df_melted, 'Probability Distribution of Latent Factors (z)', exp_path_img,dct_params)
    plot_catplot(df_melted, "Latent Factors", exp_path_img,dct_params)
    plot_swarmplot(df_melted, "Latent Factors", exp_path_img,dct_params)
    plot_violinplot(df_melted, "Latent Factors", exp_path_img,dct_params)
    plot_mce_by_latent_factor(df_mce_results.copy(), 'MCE sorted by Latent Factor', exp_path_img, dct_params) ##make a copy otherwise the original df is altered,
    plot_mce_wo_kld(df_mce_wo_kld_results.copy(), 'MCE sorted by Latent Factor - wo KLD', exp_path_img, dct_params) ##make a copy otherwise the original df is altered,
    # plot_mce_wo_kld(df_mce_wo_kld_results2.copy(), 'MCE sorted by Latent Factor - wo KLD 2', exp_path_img, dct_params) ##make a copy otherwise the original df is altered,
    plot_parallel_plot(df_mce_results.copy(), 'MCE for different Metadata', exp_path_img, dct_params)##make a copy otherwise the original df is altered,
    plot_KLD(model.ls_kld, 'KLD over Epochs (Training)', exp_path_img, dct_params)
    plot_pairplot_lf(model, 'Correlation of Latent Factors', exp_path_img, dct_params)

    # plot_mce(model, neptune_logger, max_epochs) #TODO Change method to process multiple entries
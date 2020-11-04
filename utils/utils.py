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
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import plotly.express as px
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

def create_synthetic_data_nd(no_generative_factors, experiment_path):
    no_samples = 20
    genres = ['Crime', 'Mystery', 'Thriller', 'Action', 'Drama', 'Romance','Comedy', 'War','Adventure', 'Family']
    year = [1980, 1990, 2000, 2010, 2020]
    stars = ['Tom Hanks', 'Tim Allen', 'Don Rickles','Robin Williams', 'Kirsten Dunst', 'Bonnie Hunt']
    rating = [7, 8, 9, 10]

    # dct_base_data ={'genres': genres, 'year': year, 'stars': stars, 'rating': rating}#

    ls_movies = []

    #genre-users
    ls_attributes = ['year', 'rating', 'genres', 'stars']#
    ls_attributes = ls_attributes[:no_generative_factors]
    dct_data ={'genres': genres, 'year': year, 'stars':stars, 'rating':rating}
    dct_base_data = {key:dct_data[key] for key in ls_attributes}

    n_users = 1800
    n_movies = len(ls_attributes * no_samples)
    np_user_item = np.zeros((n_users,n_movies,len(year), len(rating)),dtype="float32")

    #Create movies by sampling
    for attribute in ls_attributes:
        for i in range(no_samples):
            movie = {}
            movie[attribute] = dct_base_data[attribute][0]

            for other_attribute in ls_attributes:
                if(other_attribute == attribute):
                    continue
                if(other_attribute == 'rating' or other_attribute == 'year'):
                    movie[other_attribute] = random.choices(dct_base_data[other_attribute], k=1)[0]
                else:
                    movie[other_attribute] = random.choices(dct_base_data[other_attribute], k=2)


            ls_movies.append(movie)


    df_synthentic_data = pd.DataFrame(columns=ls_attributes, data=ls_movies)
    df_synthentic_data['id'] = df_synthentic_data.index
    df_synthentic_data.to_csv('../data/generated/syn.csv', index=False)
    df_rating = pd.get_dummies(df_synthentic_data['rating'])
    df_year = pd.get_dummies(df_synthentic_data['year'])

    ls_y = []
    no_users_with_same_preference = int(n_users / len(ls_attributes))
    #Create user history by randomly sampling
    for i in range(0, len(ls_attributes)):
        end = (i+1) * no_samples -1
        start = i * no_samples
        sr_ids = df_synthentic_data.loc[start:end]['id']
        attribute = ls_attributes[i]

        #Iterate through the total number of a user cluster
        for idx in range(no_users_with_same_preference):
            min_sample = no_samples*0.5
            max_sample = no_samples*0.7
            no_of_seen_items = int(random.uniform(min_sample,max_sample))#30, 49
            seen = random.sample(list(sr_ids.values), k=no_of_seen_items)
            user_idx = i * no_users_with_same_preference + idx

            years = df_year.iloc[seen].values
            ratings = df_rating.iloc[seen].values
            np_user_item[user_idx,seen,np.argmax(years, axis=1), np.argmax(ratings, axis=1)] = 1 #e.g.: np_user_item[user_idx,[0,1]] = [[1980,7],[1990,8]]
            ls_y.append(attribute)
        # df_synthentic_data.loc[start:end, 'y'] = ls_attributes[i]
    # print(ls_movies)


    np.random.seed(42)
    import sklearn
    np_user_item, ls_y = sklearn.utils.shuffle(np_user_item, ls_y)

    print("Shape of User-Item Matrix:{}".format(np_user_item.shape))
    return np_user_item, ls_y

def create_synthetic_3d_data(no_generative_factors, experiment_path):
    no_samples = 20
    genres = ['Crime', 'Mystery', 'Thriller', 'Action', 'Drama', 'Romance','Comedy', 'War','Adventure', 'Family']
    year = [1980, 1990, 2000, 2010, 2020]
    stars = ['Tom Hanks', 'Tim Allen', 'Don Rickles','Robin Williams', 'Kirsten Dunst', 'Bonnie Hunt']
    rating = [7, 8, 9, 10]

    # dct_base_data ={'genres': genres, 'year': year, 'stars': stars, 'rating': rating}#

    ls_movies = []

    #genre-users
    ls_attributes = ['year', 'rating', 'genres', 'stars']#
    ls_attributes = ls_attributes[:no_generative_factors]
    dct_data ={'genres': genres, 'year': year, 'stars':stars, 'rating':rating}
    dct_base_data = {key:dct_data[key] for key in ls_attributes}

    n_users = 1800
    n_movies = len(ls_attributes * no_samples)
    np_user_item = np.zeros((n_users,n_movies,2),dtype="float32")

    #Create movies by sampling
    for attribute in ls_attributes:
        for i in range(no_samples):
            movie = {}
            movie[attribute] = dct_base_data[attribute][0]

            for other_attribute in ls_attributes:
                if(other_attribute == attribute):
                    continue
                if(other_attribute == 'rating' or other_attribute == 'year'):
                    movie[other_attribute] = random.choices(dct_base_data[other_attribute], k=1)[0]
                else:
                    movie[other_attribute] = random.choices(dct_base_data[other_attribute], k=2)


            ls_movies.append(movie)

    df_synthentic_data = pd.DataFrame(columns=ls_attributes, data=ls_movies)
    df_synthentic_data['id'] = df_synthentic_data.index
    df_synthentic_data.to_csv('../data/generated/syn.csv', index=False)

    ls_y = []
    no_users_with_same_preference = int(n_users / len(ls_attributes))
    #Create user history by randomly sampling
    for i in range(0, len(ls_attributes)):
        end = (i+1) * no_samples -1
        start = i * no_samples
        sr_ids = df_synthentic_data.loc[start:end]['id']
        attribute = ls_attributes[i]

        #Iterate through the total number of a user cluster
        for idx in range(no_users_with_same_preference):
            min_sample = no_samples*0.5
            max_sample = no_samples*0.7
            no_of_seen_items = int(random.uniform(min_sample,max_sample))#30, 49
            seen = random.sample(list(sr_ids.values), k=no_of_seen_items)
            user_idx = i * no_users_with_same_preference + idx
            # df_synthentic_data.iloc[[0, 1]]['year'].values
            years = df_synthentic_data.iloc[seen]['year'].values/1000
            ratings = df_synthentic_data.iloc[seen]['rating'].values/10

            np_user_item[user_idx,seen] = list(zip(years, ratings)) #e.g.: np_user_item[user_idx,[0,1]] = [[1980,7],[1990,8]]
            ls_y.append(attribute)
        # df_synthentic_data.loc[start:end, 'y'] = ls_attributes[i]
    # print(ls_movies)


    np.random.seed(42)
    import sklearn
    np_user_item, ls_y = sklearn.utils.shuffle(np_user_item, ls_y)

    print("Shape of User-Item Matrix:{}".format(np_user_item.shape))
    return np_user_item, ls_y

def create_synthetic_data(no_generative_factors, experiment_path, expanded_user_item):
    if(expanded_user_item):
        # return create_synthetic_3d_data()
        return create_synthetic_data_nd(no_generative_factors, experiment_path)

    return create_synthetic_data_simple(no_generative_factors, experiment_path)

def create_synthetic_data_simple(no_generative_factors, experiment_path):
    no_samples = 20
    genres = ['Crime', 'Mystery', 'Thriller', 'Action', 'Drama', 'Romance','Comedy', 'War','Adventure', 'Family']
    year = ['1980', '1990', '2000', '2010', '2020']
    stars = ['Tom Hanks', 'Tim Allen', 'Don Rickles','Robin Williams', 'Kirsten Dunst', 'Bonnie Hunt']
    rating = ['7', '8', '9', '10']

    # dct_base_data ={'genres': genres, 'year': year, 'stars': stars, 'rating': rating}#

    ls_movies = []

    #genre-users
    ls_attributes = ['year', 'rating', 'genres', 'stars']#
    ls_attributes = ls_attributes[:no_generative_factors]
    dct_data = {'genres': genres, 'year': year, 'stars': stars, 'rating': rating}
    dct_base_data = {key: dct_data[key] for key in ls_attributes}

    n_users = 1800
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


    df_synthentic_data = pd.DataFrame(columns=ls_attributes, data=ls_movies)
    df_synthentic_data['id'] = df_synthentic_data.index
    df_synthentic_data.to_csv('../data/generated/syn.csv', index=False)


    ls_y = []
    no_users_with_same_preference = int(n_users / len(ls_attributes))
    for i in range(0, len(ls_attributes)):
        end = (i+1) * no_samples -1
        start = i * no_samples
        sr_ids = df_synthentic_data.loc[start:end]['id']
        attribute = ls_attributes[i]
        #

        for idx in range(no_users_with_same_preference):
            min_sample = no_samples*0.5
            max_sample = no_samples*0.7
            no_of_seen_items = int(random.uniform(min_sample,max_sample))#30, 49
            seen = random.sample(list(sr_ids.values), k=no_of_seen_items)
            user_idx = i * no_users_with_same_preference + idx
            np_user_item[user_idx,seen] = 1
            ls_y.append(attribute)
        # df_synthentic_data.loc[start:end, 'y'] = ls_attributes[i]
    # print(ls_movies)
    create_heatmap(np_arr = np_user_item,
                   title = "Heatmap of User-Item Matrix",
                   y_label="User ID",
                   x_label="Movie ID",
                   experiment_path = experiment_path,
                   dct_params= {'sample_size':no_samples, 'no_users': n_users})

    np.random.seed(42)
    import sklearn
    np_user_item, ls_y = sklearn.utils.shuffle(np_user_item, ls_y)

    print("Shape of User-Item Matrix:{}".format(np_user_item.shape))
    return np_user_item, ls_y


def create_heatmap(np_arr, title, y_label, x_label, experiment_path, dct_params):
    ax = sns.heatmap(np_arr)
    # plt.suptitle(title)
    ax.set(title=title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    save_figure(ax.get_figure(), experiment_path, 'heatmap-user-item', dct_params)
    plt.show()




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
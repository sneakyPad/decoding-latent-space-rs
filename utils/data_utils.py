import torch
import json
import seaborn as sns
import pandas as pd
from sklearn import decomposition
from sklearn.metrics import mean_squared_error
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
from utils import plot_utils, utils

def generate_distribution_df():
    dct_attribute_distribution = utils.compute_relative_frequency(
        pd.read_csv('../data/generated/syn.csv'))
    utils.save_dict_as_json(dct_attribute_distribution, 'syn_attribute_distribution.json')


def create_synthetic_data_nd(no_generative_factors, experiment_path):
    no_samples = 20
    genres = ['Crime', 'Mystery', 'Thriller', 'Action']
    year = [1980, 1990, 2000, 2010]
    stars = ['Tom Hanks', 'Tim Allen', 'Don Rickles','Robin Williams']
    rating = [7, 8, 9, 10]

    # dct_base_data ={'genres': genres, 'year': year, 'stars': stars, 'rating': rating}#

    ls_movies = []

    #genre-users
    ls_attributes = ['year', 'rating', 'genres', 'stars']#
    ls_attributes = ls_attributes[:no_generative_factors]
    dct_data ={'genres': genres, 'year': year, 'stars':stars, 'rating':rating}
    dct_base_data = {key:dct_data[key] for key in ls_attributes}

    n_users = 2400*4
    n_movies = len(ls_attributes * no_samples)
    if(no_generative_factors==2):
        np_user_item = np.zeros((n_users,n_movies,len(year), len(rating)),dtype="float32")
    else:
        np_user_item = np.zeros((n_users,n_movies,len(year), len(rating), len(genres)),dtype="float32")

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
                    movie[other_attribute] = random.choices(dct_base_data[other_attribute], k=1)[0] #k= number of samples


            ls_movies.append(movie)


    df_synthentic_data = pd.DataFrame(columns=ls_attributes, data=ls_movies)
    df_synthentic_data['id'] = df_synthentic_data.index
    df_synthentic_data.to_csv('../data/generated/syn.csv', index=False)
    df_rating = pd.get_dummies(df_synthentic_data['rating'])
    df_year = pd.get_dummies(df_synthentic_data['year'])
    df_genres = pd.get_dummies(df_synthentic_data['genres'])

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
            genres = df_genres.iloc[seen].values
            if(no_generative_factors == 2):
                np_user_item[user_idx, seen, np.argmax(years, axis=1), np.argmax(ratings, axis=1)] = 1  # e.g.: np_user_item[user_idx,[0,1]] = [[1980,7],[1990,8]]

            else:
                np_user_item[user_idx, seen, np.argmax(years, axis=1), np.argmax(ratings, axis=1), np.argmax(genres,axis=1)] = 1  # e.g.: np_user_item[user_idx,[0,1]] = [[1980,7],[1990,8]]

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

def create_synthetic_data(no_generative_factors, experiment_path, expanded_user_item, continous_data, normalvariate, noise):
    if(expanded_user_item):
        # return create_synthetic_3d_data()
        return create_synthetic_data_nd(no_generative_factors, experiment_path)

    return create_synthetic_data_simple(no_generative_factors, experiment_path, continous_data, normalvariate, noise)

def create_synthetic_data_simple(no_generative_factors, experiment_path, continous_data, normalvariate, noise):

    no_samples = 20
    genres = ['Crime', 'Mystery', 'Thriller', 'Action', 'Drama']#, 'Romance','Comedy', 'War','Adventure', 'Family'
    year = ['1980', '1990', '2000', '2010', '2020']
    stars = ['Tom Hanks', 'Tim Allen', 'Don Rickles','Robin Williams', 'Kirsten Dunst']#, 'Bonnie Hunt'
    rating = ['6','7', '8', '9', '10']

    # dct_base_data ={'genres': genres, 'year': year, 'stars': stars, 'rating': rating}#

    ls_movies = []

    #genre-users
    ls_attributes = ['year', 'rating', 'genres', 'stars']#
    ls_attributes = ls_attributes[:no_generative_factors]
    dct_data = {'genres': genres, 'year': year, 'stars': stars, 'rating': rating}
    dct_base_data = {key: dct_data[key] for key in ls_attributes}

    n_users = 600*30
    n_movies = len(ls_attributes * no_samples)
    np_user_item = np.zeros((n_users,n_movies),dtype="float32")

    # Create movies by sampling
    for attribute in ls_attributes:
        for i in range(no_samples):
            movie = {}
            movie[attribute] = dct_base_data[attribute][0]

            for other_attribute in ls_attributes:
                if (other_attribute == attribute):
                    continue
                if (other_attribute == 'rating' or other_attribute == 'year'):
                    movie[other_attribute] = random.choices(dct_base_data[other_attribute], k=1)[0]
                else:
                    movie[other_attribute] = random.choices(dct_base_data[other_attribute], k=1)[
                        0]  # k= number of samples

            ls_movies.append(movie)


    df_synthentic_data = pd.DataFrame(columns=ls_attributes, data=ls_movies)
    df_synthentic_data['id'] = df_synthentic_data.index
    df_synthentic_data.to_csv('../data/generated/syn.csv', index=False)


    ls_y = []
    no_users_with_same_preference = int(n_users / len(ls_attributes))
    for i in range(0, len(ls_attributes)):
        if(noise):
            noise_val = 5
            end = (i+1) * no_samples -1
            end = min(n_movies, end + noise_val )
            start = i * no_samples
            start = max(0, start - noise_val)
        else:
            end = (i + 1) * no_samples - 1
            start = i * no_samples
        sr_ids = df_synthentic_data.loc[start:end]['id']
        attribute = ls_attributes[i]
        #

        for idx in range(no_users_with_same_preference):
            min_sample = int(no_samples*0.2)
            max_sample = int(no_samples*0.4)
            no_of_seen_items = int(random.uniform(min_sample,max_sample))#30, 49
            if(normalvariate):
                mu = (end+1)-(no_samples/2)
                sigma = 2
                seen = [int(random.normalvariate(mu, sigma)) for k in range(0,no_of_seen_items)]
                # seen = random.normalvariate(mu, sigma)
            else:
                seen = random.sample(list(sr_ids.values), k=no_of_seen_items)
            user_idx = i * no_users_with_same_preference + idx
            if(continous_data):
                # np_user_item[user_idx,seen] = 1* random.uniform(0,0.33)*i #This creates ranges of 0-1/3, 1/3-2/3,2/3-1
                np_user_item[user_idx,seen] = random.uniform(0.33, 1)#*i
            else:
                np_user_item[user_idx,seen] = 1
            ls_y.append(attribute)
        # df_synthentic_data.loc[start:end, 'y'] = ls_attributes[i]
    # print(ls_movies)
    plot_utils.create_heatmap(np_arr = np_user_item,
                   title = "Heatmap of User-Item Matrix",
                   y_label="User ID",
                   x_label="Movie ID",
                   file_name="heatmap-user-item",
                   experiment_path = experiment_path+"images/",
                   dct_params= {'sample_size':no_samples, 'no_users': n_users})

    np.random.seed(42)
    import sklearn
    np_user_item, ls_y = sklearn.utils.shuffle(np_user_item, ls_y)

    plot_utils.create_heatmap(np_arr=np_user_item,
                              title="Heatmap of shuffled User-Item Matrix",
                              y_label="User ID",
                              x_label="Movie ID",
                              file_name="shuffled-heatmap-user-item",
                              experiment_path=experiment_path+"images/",
                              dct_params={'sample_size': no_samples, 'no_users': n_users})

    print("Shape of User-Item Matrix:{}".format(np_user_item.shape))
    return np_user_item, ls_y

def extract_rating_information(df, hack):
    user_item_mx_offset = 0 #TODO This is rather a hack, consider to remove it
    if(hack):
        user_item_mx_offset = 1 #needed by the manual_create_user_item_matrix()
    unique_movies = len(df["movieId"].unique()) + user_item_mx_offset
    max_unique_movies = df["movieId"].unique().max() + user_item_mx_offset
    ls_users = df["userId"].unique()
    unique_users = len(df["userId"].unique()) + user_item_mx_offset
    print('Unique movies: {}\nMax unique movies:{}\nUnique users: {}'.format(unique_movies, max_unique_movies, unique_users))

    return unique_movies, max_unique_movies, ls_users, unique_users

def pivot_create_user_item_matrix(df: pd.DataFrame, simplified_rating: bool):
    print('---- Create User Item Matrix: Pivot Style----')
    dct_index2itemId={}
    unique_movies, max_unique_movies, ls_users, unique_users = extract_rating_information(df, False)

    if(simplified_rating):
        df['rating'] = 1
    else:
        df['rating'] = df['rating']/5
    df_user_item = df.pivot(index="userId", columns="movieId", values="rating")
    df_user_item = df_user_item.fillna(0)
    ##Create Mapping
    # dct_index2itemId ={}
    for index, item_id in enumerate(df_user_item.columns):
        dct_index2itemId[index]=item_id

    np_user_item = df_user_item.to_numpy()
    print('Shape of Matrix:{}'.format(np_user_item.shape))
    print('Stucture of the matrix: \n ______| movie_1 | movie_2 | ... | movie_n \n user_1| \n user_2| \n ... \n user_m|')

    return np_user_item.astype(np.float32), unique_movies, max_unique_movies, dct_index2itemId


def manual_create_user_item_matrix(df, simplified_rating: bool):
    print('---- Create User Item Matrix: Manual Style ----')

    unique_movies, max_unique_movies, ls_users, unique_users = extract_rating_information(df, True)
    np_user_item_mx = np.zeros((unique_users, max_unique_movies), dtype=np.float_) #type: nd_array[[]]
    print('Shape of Matrix:{}'.format(np_user_item_mx.shape))

    print('Fill user-item matrix...')
    print('Stucture of the matrix: \n ______| movie_1 | movie_2 | ... | movie_n \n user_1| \n user_2| \n ... \n user_m|')
    for user_id in ls_users:
        # if(user_id%100 == 0):
        #     print("User-Id: {} ".format(user_id))
        ls_seen_items = df.loc[df["userId"] == user_id]["movieId"].values
        if(simplified_rating):
            np_user_item_mx[user_id][ls_seen_items] = 1 #TODO what happens to row 0?
        else:
            #Normalize values
            from sklearn import preprocessing
            from sklearn.preprocessing import normalize

            # min_max_scaler = preprocessing.MinMaxScaler()

            ls_rated_items = df.loc[df["userId"] == user_id]["rating"].values
            # df_array = ls_rated_items.reshape(-1, 1)
            # ls_normalized_ratings = preprocessing.normalize(df_array, axis=0)

            # ls_normalized_ratings = normalize(ls_rated_items, norm='max', axis=1)
            # ls_normalized_ratings = min_max_scaler.fit_transform(ls_rated_items)
            ls_normalized_ratings = [float(i) / max(ls_rated_items) for i in ls_rated_items]
            # ls_normalized_ratings = (ls_rated_items - ls_rated_items.min()) / (ls_rated_items.max() - ls_rated_items.min())
            np_user_item_mx[user_id][ls_seen_items] = ls_normalized_ratings

    return np_user_item_mx.astype(np.float32), max_unique_movies

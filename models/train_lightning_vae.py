# pip install pytorch-lightning
# pip install neptune-client
#%%
from __future__ import print_function
import wandb
from pytorch_lightning.loggers import WandbLogger


import torch, torch.nn as nn, torchvision, torch.optim as optim
import numpy as np

import pandas as pd
from tqdm import tqdm
from pytorch_lightning.loggers.neptune import NeptuneLogger

from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
# import recmetrics
# from surprise import Reader, SVD, Dataset
# from surprise.model_selection import train_test_split
import ast
from collections import defaultdict

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import utils.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn import manifold, decomposition

import pickle
import wandb

from datetime import datetime
import os
#ToDo EDA:
# - Long Tail graphics
# - Remove user who had less than a threshold of seen items
# - Create Markdown with EDA results

#ToDo input_params:
# Parameter that should be tweakable by invoking the routine:
# - epochs
# - learning_rate
# - batch_size
# - simplified_rating
# - hidden_layer number
# - Algorithm: VAE, AE or SVD

#ToDo metrics:
# Add https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093

#ToDo training:
# Add test loss


seed = 42
torch.manual_seed(seed)


##This method creates a user-item matrix by transforming the seen items to 1 and adding unseen items as 0 if simplified_rating is set to True
##If set to False, the actual rating is taken
##Shape: (n_user, n_items)
max_unique_movies = 0
unique_movies = 0
dct_index2itemId={}
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
    global max_unique_movies
    global unique_movies
    unique_movies, max_unique_movies, ls_users, unique_users = extract_rating_information(df, False)

    if(simplified_rating):
        df['rating'] = 1

    df_user_item = df.pivot(index="userId", columns="movieId", values="rating")
    df_user_item = df_user_item.fillna(0)
    ##Create Mapping
    # dct_index2itemId ={}
    for index, item_id in enumerate(df_user_item.columns):
        dct_index2itemId[index]=item_id

    np_user_item = df_user_item.to_numpy()
    print('Shape of Matrix:{}'.format(np_user_item.shape))
    print('Stucture of the matrix: \n ______| movie_1 | movie_2 | ... | movie_n \n user_1| \n user_2| \n ... \n user_m|')

    return np_user_item.astype(np.float32), unique_movies


def manual_create_user_item_matrix(df, simplified_rating: bool):
    print('---- Create User Item Matrix: Manual Style ----')
    global max_unique_movies
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


def generate_mask(ts_batch_user_features, tsls_yhat_user, user_based_items_filter: bool):
    # user_based_items_filter == True is what most people do
    mask = None
    if (user_based_items_filter):
        mask = ts_batch_user_features == 0.  # filter out everything except what the user has seen , mask_zeros
    else:
        # TODO Mask filters also 1 out, that's bad
        mask = ts_batch_user_features == tsls_yhat_user  # Obtain a mask for filtering out items that haven't been seen nor recommended, basically filter out what is 0:0 or 1:1
    return mask

class VAE(pl.LightningModule):
    def __init__(self, conf:dict, *args, **kwargs):
        super().__init__()

        # self.kwargs = kwargs
        self.save_hyperparameters(conf)

        self.beta = self.hparams["beta"]
        self.avg_mce = 0.0
        self.train_dataset = None
        self.test_dataset = None
        self.test_size = self.hparams["test_size"]
        self.no_latent_factors = self.hparams["latent_dim"]
        self.max_unique_movies = 0
        self.unique_movies =0
        self.np_user_item = None
        self.small_dataset = self.hparams["small_dataset"]
        self.simplified_rating = self.hparams["simplified_rating"]
        self.max_epochs = self.hparams["max_epochs"]
        self.load_dataset() #additionaly assigns self.unique_movies and self.np_user_item

        #nn.Linear layer creates a linear function (θx + b), with its parameters initialized
        self.fc1 = nn.Linear(in_features=self.unique_movies, out_features=400) #input
        self.fc21 = nn.Linear(in_features=400, out_features=self.no_latent_factors) #encoder mean
        self.fc22 = nn.Linear(in_features=400, out_features=self.no_latent_factors) #encoder variance
        self.fc3 = nn.Linear(in_features=self.no_latent_factors, out_features=400) #hidden layer
        self.fc4 = nn.Linear(in_features=400, out_features=self.unique_movies)


        self.z = None
        self.np_z_test = np.empty((0, self.no_latent_factors))#self.test_dataset.shape[0]
        self.np_mu_test = np.empty((0, self.no_latent_factors))
        self.np_logvar_test = np.empty((0, self.no_latent_factors))

        self.np_z_train = np.empty((0, self.no_latent_factors))  # self.test_dataset.shape[0]
        self.np_mu_train = np.empty((0, self.no_latent_factors))
        self.np_logvar_train = np.empty((0, self.no_latent_factors))

        # if (kwargs.get('load_saved_attributes') == True):
        #     dct_attributes = self.load_attributes(kwargs.get('saved_attributes_path'))
        #     print('attributes loaded')
        #
        #     self.np_z_train = dct_attributes['np_z_train']

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def compute_z(self, mu, logvar):

        z = self.reparameterize(mu, logvar)
        return z

    def forward(self, x, **kwargs):
        #Si

        if(kwargs):
            z = kwargs['z']
            mu = kwargs['mu']
            logvar = kwargs['logvar']
        else:
            mu, logvar = self.encode(x.view(-1, self.unique_movies))
            z = self.compute_z(mu, logvar)
            self.z = z

        return self.decode(z), mu, logvar

    def load_dataset(self):
        if (self.small_dataset):
            print("Load small dataset of ratings.csv")
            df_ratings = pd.read_csv("../data/movielens/small/ratings.csv")

        else:
            print("Load large dataset of ratings.csv")
            df_ratings = pd.read_csv("../data/movielens/large/ratings.csv")

        print('Shape of dataset:{}'.format(df_ratings.shape))
        self.np_user_item, self.unique_movies = pivot_create_user_item_matrix(df_ratings,True)#manual_create_user_item_matrix(df_ratings, simplified_rating=self.simplified_rating)
        # self.np_user_item, self.max_unique_movies = manual_create_user_item_matrix(df_ratings, simplified_rating=self.simplified_rating)
        self.train_dataset, self.test_dataset = train_test_split(self.np_user_item, test_size=self.test_size, random_state=42)

    def train_dataloader(self):
        #TODO Change shuffle to True, just for dev purpose switched on
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory=True
        )
        return train_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=32, shuffle=False, num_workers=1
        )
        return test_loader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.MSELoss()  # mean-squared error loss
        # scheduler = StepLR(optimizer, step_size=1)
        return optimizer#, scheduler

    def training_step(self, batch, batch_idx):

        print('train step')
        ts_batch_user_features = batch
        recon_batch, ts_mu_chunk, ts_logvar_chunk = self.forward(ts_batch_user_features)  # sample data
        if(self.current_epoch == self.max_epochs-1):
            print("Last rounf yipi")
            ls_grad_z = self.compute_z(ts_mu_chunk, ts_logvar_chunk)
            self.np_z_train = np.append(self.np_z_train, np.asarray(ls_grad_z.tolist()), axis=0) #TODO Describe in thesis that I get back a grad object instead of a pure tensor as it is in the test method since we are in the training method.
            self.np_mu_train = np.append(self.np_mu_train, np.asarray(ts_mu_chunk.tolist()), axis=0)
            self.np_logvar_train = np.append(self.np_logvar_train, np.asarray(ts_logvar_chunk.tolist()), axis=0)



        batch_loss = loss_function(recon_batch, ts_batch_user_features, ts_mu_chunk, ts_logvar_chunk, self.beta, unique_movies)
        loss = batch_loss/len(ts_batch_user_features)

        # tensorboard_logs = {'train_loss': loss}
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}



    # def validation_step(self, batch, batch_idx):
    #     return 0
    #
    def test_step(self, batch, batch_idx):
        batch_mce =0
        #TODO needs to be outside of test loop
        print('Shape np_z_train: {}'.format(self.np_z_train.shape))
        self.z_mean_train = self.np_z_train.mean(axis=0)
        self.z_min_train = self.np_z_train.min(axis=0)
        self.z_max_train = self.np_z_train.max(axis=0)

        print('test step')

        # self.eval()
        ts_batch_user_features = batch

        test_loss = 0
        # with torch.no_grad():
        #     for i, data in enumerate(test_loader):

        recon_batch, ts_mu_chunk, ts_logvar_chunk = self(ts_batch_user_features)
        ls_z = self.compute_z(ts_mu_chunk, ts_logvar_chunk)

        self.np_z_test = np.append(self.np_z_test, np.asarray(ls_z), axis=0) #TODO get rid of np_z_chunk and use np.asarray(mu_chunk)
        self.np_mu_test = np.append(self.np_mu_test, np.asarray(ts_mu_chunk), axis =0)
        self.np_logvar_test = np.append(self.np_logvar_test, np.asarray(ts_logvar_chunk), axis =0)
        # self.np_z = np.vstack((self.np_z, np_z_chunk))


        batch_rmse, batch_mse, batch_rmse_wo_zeros, batch_mse_wo_zeros = self.calculate_batch_metrics(recon_batch=recon_batch, ts_batch_user_features =ts_batch_user_features)
        batch_loss = loss_function(recon_batch, ts_batch_user_features, ts_mu_chunk, ts_logvar_chunk, self.beta, unique_movies).item()
        # batch_mce = mce_batch(model, ts_batch_user_features, k=1)

        #to be rermoved mean_mce = { for single_mce in batch_mce}
        loss = batch_loss / len(ts_batch_user_features)

        return {'test_loss': batch_loss,
                'mce': batch_mce,
                'rmse': batch_rmse,
                'mse': batch_mse,
                'rmse_wo_zeros': batch_rmse_wo_zeros,
                'mse_wo_zeros': batch_mse_wo_zeros
                }

        # test_loss /= len(test_loader.dataset)
        # print('====> Test set loss: {:.4f}'.format(test_loss))

    def test_epoch_end(self, outputs):
        # avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_loss = np.array([x['test_loss'] for x in outputs]).mean()

        # ls_mce = {x['mce'] for x in outputs}
        utils.save_dict_as_json(outputs[0]['mce'], 'mce_results.json')
        # avg_mce = dict(calculate_mean_of_ls_dict(ls_mce))

        avg_rmse = np.array([x['rmse'] for x in outputs]).mean()
        avg_rmse_wo_zeros = np.array([x['rmse_wo_zeros'] for x in outputs]).mean()
        avg_mse = np.array([x['mse'] for x in outputs]).mean()
        avg_mse_wo_zeros = np.array([x['mse_wo_zeros'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss}
        wandb_logger.log_metrics({'rmse': avg_rmse, 'rmse_wo_zeros':avg_mse_wo_zeros})
        # neptune_logger.experiment.log_metric('rmse', avg_rmse)
        # neptune_logger.experiment.log_metric('rmse_wo_zeros', avg_rmse_wo_zeros)
        # neptune_logger.experiment.log_metric('mse', avg_mse)
        # neptune_logger.experiment.log_metric('mse_wo_zeros', avg_mse_wo_zeros)

        # self.avg_mce = avg_mce

        return {'test_loss': avg_loss, 'log': tensorboard_logs, 'rmse': avg_rmse}#, , 'mce':avg_mce



    def calculate_batch_metrics(self, recon_batch, ts_batch_user_features):
        # Compute MSE
        # TODO MOre generic ...

        # mask = generate_mask(ts_batch_user_features, tsls_yhat_user, user_based_items_filter=loss_user_items_only)
        # tsls_yhat_user_filtered = tsls_yhat_user[~mask]  # Predicted: Filter out unseen+unrecommended items
        # ts_user_features_seen = ts_batch_user_features[~mask]  # Ground Truth: Filter out unseen+unrecommended items

        # TODO ...than this approach

        batch_rmse = 0
        batch_mse = 0
        batch_rmse_wo_zeros = 0
        batch_mse_wo_zeros = 0
        ls_yhat_user = recon_batch * ts_batch_user_features  # Set all items to zero that are of no interest and haven't been seen
        for idx, tensor in enumerate(ls_yhat_user):
            np_y = ts_batch_user_features[idx].data.numpy()
            np_y_wo_zeros = np_y[np.nonzero(np_y)]  # inner returns the index

            np_yhat = tensor.data.numpy()
            np_yhat_wo_zeros = np_yhat[np.nonzero(np_yhat)]

            rmse, mse = utils.calculate_metrics(np_y, np_yhat)
            rmse_wo_zeros, mse_wo_zeros = utils.calculate_metrics(np_y_wo_zeros, np_yhat_wo_zeros)
            batch_rmse += rmse
            batch_rmse_wo_zeros += rmse_wo_zeros

            batch_mse += mse
            batch_mse_wo_zeros += mse_wo_zeros
        # batch_rmse, batch_mse = utils.calculate_metrics(ts_batch_user_features,ls_yhat_user)
        avg_rmse = batch_rmse / ls_yhat_user.shape[0]
        avg_rmse_wo_zeros = batch_rmse_wo_zeros / ls_yhat_user.shape[0]

        avg_mse = batch_mse / ls_yhat_user.shape[0]
        avg_mse_wo_zeros = batch_mse_wo_zeros / ls_yhat_user.shape[0]
        return avg_rmse, avg_mse, avg_rmse_wo_zeros, avg_mse_wo_zeros

    def load_attributes(self, path): #'filename.pickle'
        with open(path, 'rb') as handle:
            dct_attributes = pickle.load(handle)
        self.np_z_train = dct_attributes['np_z_train']
        print('Attributes loaded')

    def save_attributes(self, path):
        dct_attributes = {'np_z_train':self.np_z_train}
        with open(path, 'wb') as handle:
            pickle.dump(dct_attributes, handle)
        print('Attributes saved')


def my_eval(expression):
    try:
        return ast.literal_eval(str(expression))
    except SyntaxError: #e.g. a ":" or "(", which is interpreted by eval as command
            return [expression]
    except ValueError: #e.g. an entry is nan, in that case just return an empty string
        return ''

def mce_relative_frequency(y_hat, y_hat_latent):
    dct_attribute_distribution = utils.load_json_as_dict('attribute_distribution.json') #    load relative frequency distributioon from dictionary (pickle it)
    # dct_dist = pickle.load(movies_distribution)

    dct_mce = defaultdict(float)
    for idx_vector in range(y_hat.shape[0]):
        for attribute in y_hat:
            if(attribute not in ['Unnamed: 0', 'unnamed_0', 'plot_outline']):
                ls_y_attribute_val = my_eval(y_hat.iloc[idx_vector][attribute]) #e.g. Stars: ['Pitt', 'Damon', 'Jolie']
                ls_y_latent_attribute_val = my_eval(y_hat_latent.iloc[idx_vector][attribute]) #e.g Stars: ['Depp', 'Jolie']
                mean = 0
                cnt_same = 0
                mce=0
                try:
                    #Two cases: Either cell contains multiple values, than it is a list
                    #or it contains a single but not in a list. In that case put it in a list
                    if(type(ls_y_latent_attribute_val) is not list):
                        ls_y_latent_attribute_val = [ls_y_latent_attribute_val]
                        ls_y_attribute_val =[ls_y_attribute_val]

                    #Go through elements of a cell
                    for value in ls_y_latent_attribute_val: #same as characteristic
                        if(value in ls_y_attribute_val):
                            # mean += 1
                            mce +=1
                            # ls_y_latent_attribute_val.pop(value)

                        else:
                            # characteristic = y_hat.loc[idx_vector, attribute]
                            relative_frequency = dct_attribute_distribution[attribute]['relative'][str(value)]
                            mce += relative_frequency
                            # print('\t Value: {}, Relative frequency:{}'.format(value, relative_frequency))
                    #if no values are presented in the current cell than assign highest error
                    if(len(ls_y_latent_attribute_val)==0):
                        mce =1
                    else:
                        mce = mce/len(ls_y_latent_attribute_val)
                    # print('Attribute: {}, mce:{}'.format(attribute, mce))

                    dct_mce[attribute] = mce
                except (KeyError, TypeError, ZeroDivisionError) as e:
                    print("Error Value:{}".format(value))

    return dct_mce

def calculate_mean_of_ls_dict(ls_dict: list):
    dct_sum = defaultdict(float)

    for dict in ls_dict:
        for key, val in dict.items():
            dct_sum[key] += val
    np_mean_vals = np.array(list(dct_sum.values())) / len(ls_dict)
    dct_mean = list(zip(dct_sum.keys(), np_mean_vals))
    print(dct_mean)
    return dct_mean

def match_metadata(indezes, df_links):
    # ls_indezes = y_hat.values.index
    #TODO Source read_csv out
    df_links = pd.read_csv('../data/movielens/small/links.csv')
    df_movies = pd.read_csv('../data/generated/df_movies_cleaned3.csv')
    # df_imdb_ids = df_links.loc[df_links['movieId'].isin(indezes),['imdbId']] #dataframe

    global dct_index2itemId

    ls_ml_ids = [dct_index2itemId[matrix_index] for matrix_index in indezes] #ml = MovieLens


    sr_imdb_ids = df_links[df_links["movieId"].isin(ls_ml_ids)]['imdbId'] #If I want to keep the
    imdb_ids = sr_imdb_ids.array

    # print('no of imdbIds:{}, no of indezes:{}'.format(len(imdb_ids), len(indezes)))
    #TODO Fill df_movies with MovieLensId or download large links.csv
    if(len(imdb_ids) < len(indezes)):
        print('There were items recommended that have not been seen by any users in the dataset. Trained on 9725 movies but 193610 are available so df_links has only 9725 to pick from')
    assert len(imdb_ids) == len(indezes)
    #df_links.loc[df_links["movieId"] == indezes]
    df_w_metadata = df_movies.loc[df_movies['imdbid'].isin(imdb_ids)]

    return df_w_metadata


def alter_z(ts_z, latent_factor_position, model):
    ts_z[:, latent_factor_position] = model.z_max_train[latent_factor_position] #32x no_latent_factors, so by accesing [:,pos] I get the latent factor for the batch of 32 users
    return ts_z

#MCE is calculated for each category
def mce_batch(model, ts_batch_features, k=0):
    dct_mce_mean = defaultdict()
    # hold n neurons of hidden layer
    # change 1 neuron
    ls_y_hat, mu, logvar = model(ts_batch_features)
    z = model.z
    for latent_factor_position in range(model.no_latent_factors):
        print("Calculate MCEs for position: {} in vector z".format(latent_factor_position))
        ts_altered_z = alter_z(z, latent_factor_position, model)

        ls_y_hat_latent_changed, mu, logvar = model(ts_batch_features, z=ts_altered_z, mu=mu,logvar=logvar)
        # mce()
        df_links = None
        ls_idx_y = (-ts_batch_features).argsort()
        ls_idx_yhat = (-ls_y_hat).argsort() # argsort returns indices of the given list in ascending order. For descending we invert the list, so each element is inverted
        ls_idx_yhat_latent = (-ls_y_hat_latent_changed).argsort()  # argsort returns indices of the given list in ascending order. For descending we invert the list, so each element is inverted
        mask_not_null = np.zeros(ts_batch_features.shape, dtype=bool)
        ls_indizes_not_null = torch.nonzero(ts_batch_features, as_tuple=False)


        #Collect the index of the items that were seen
        ls_non_zeroes = torch.nonzero(ls_idx_yhat, as_tuple=True)#ts_batch_features
        tpl_users = ls_non_zeroes[0]
        tpl_items = ls_non_zeroes[1]
        dct_seen_items = defaultdict(list)
        for idx in range(0,len(tpl_users)):
            user_idx = int(tpl_users[idx])
            item_idx = int(tpl_items[idx])
            dct_seen_items[user_idx].append(item_idx)
            # print(dct_seen_items)

        #TODO Can be slow, find another way
        # for user_idx, item_idx in ls_indizes_not_null:
        #     mask_not_null[user_idx][item_idx] = True
        # mask_not_null = ts_batch_features > 0 TODO This is an alternative to the for loop, test it


        #Go through the list of list of the predicted batch
        # for user_idx, ls_seen_items in dct_seen_items.items(): #ls_seen_items = ls_item_vec
        ls_dct_mce = []
        for user_idx in tqdm(range(len(ls_idx_yhat)), total = len(ls_idx_yhat)):
            y_hat = ls_y_hat[user_idx]
            y_hat_latent = ls_y_hat_latent_changed[user_idx]

            if(k == 0):
                k = 10#len(ls_seen_items) #no_of_seen_items  TODO Add k

            y_hat_k_highest = (-y_hat).argsort()[:k] #Alternative: (-y_hat).sort().indices[:no_of_seen_items]
            y_hat_latent_k_highest = (-y_hat_latent).argsort()[:k] #Alternative: (-y_hat).sort().indices[:no_of_seen_items]

            y_hat_w_metadata = match_metadata(y_hat_k_highest.tolist(), df_links)
            y_hat_latent_w_metadata = match_metadata(y_hat_latent_k_highest.tolist(), df_links)

            single_mce = mce_relative_frequency(y_hat_w_metadata, y_hat_latent_w_metadata) #mce for n columns
            ls_dct_mce.append(single_mce)

            # print(single_mce)
        dct_mce_mean[latent_factor_position] = dict(calculate_mean_of_ls_dict(ls_dct_mce))
    return dct_mce_mean



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta, unique_movies):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, unique_movies), reduction='sum') #TODO: Is that correct? binary cross entropy - (Encoder)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #Kullback Leibler (Decoder)

    return BCE + beta *KLD #beta = disentangle factor



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

if __name__ == '__main__':

    train_dataset = None
    test_dataset = None
    max_epochs = 5

    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--max_epochs', type=int, default=max_epochs, metavar='N',
                        help='number of max epochs to train (default: 15)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=0, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()


    torch.manual_seed(100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #  use gpu if available

    #%%
    model_params = {"simplified_rating": True,
                    "small_dataset": True,
                    "test_size": 0.2,#TODO Change test size to 0.33
                    "latent_dim":3,
                    "beta":1,
                    "max_epochs": max_epochs}
    # model_params.update(args.__dict__)
    # print(**model_params)

    merged_params = (lambda first_dict, second_dict: {**first_dict, **second_dict})(args.__dict__, model_params)
    # print(merged_params)



    wandb_logger = WandbLogger(project='recommender-xai', tags=['vae'])


    #%%
    train = False
    base_path = 'results/models/vae/'

    ls_epochs = [5]
    ls_latent_factors = [3]
    ls_disentangle_factors = [1] #TODO: Maybe try out 10 here


    # ls_epochs = [50, 500, 2000]
    # ls_latent_factors = [3, 5, 10, 15]

    for epoch in ls_epochs:
        for lf in ls_latent_factors:
            for beta in ls_disentangle_factors:
                print("Processing model with: {} epochs, {} latent factors, {} beta".format(epoch, lf, beta))
                exp_name = "{}_beta_{}_epochs_{}_lf".format(beta,epoch,lf)
                model_name = exp_name+".ckpt"
                attribute_name = exp_name+"_attributes.pickle"
                model_path = base_path + model_name
                attribute_path = base_path + attribute_name

                model_params['max_epochs'] = epoch
                model_params['latent_dim'] = lf
                model_params['beta'] = beta

                args.max_epochs = epoch
                trainer = pl.Trainer.from_argparse_args(args,
                                                        logger=wandb_logger, #False
                                                        gpus=0,
                                                        weights_summary='full',
                                                        checkpoint_callback = False,
                )

                if(train):
                    print('<---------------------------------- VAE Training ---------------------------------->')
                    print("Running with the following configuration: \n{}".format(args))
                    model = VAE(model_params)
                    utils.print_nn_summary(model)

                    print('------ Start Training ------')
                    trainer.fit(model)

                    print('------ Saving model ------')
                    trainer.save_checkpoint(model_path)
                    model.save_attributes(attribute_path)

                print('------ Load model -------')
                test_model = VAE.load_from_checkpoint(model_path)#, load_saved_attributes=True, saved_attributes_path='attributes.pickle'
                test_model.load_attributes(attribute_path)

                # print("show np_z_train mean:{}, min:{}, max:{}".format(z_mean_train, z_min_train, z_max_train ))
                print('------ Start Test ------')
                trainer.test(test_model) #The test loop will not be used until you call.
                # print(results)

                # %%
                dct_param ={'epochs':epoch, 'lf':lf,'beta':beta}
                experiment_path = create_experiment_directory()

                utils.plot_results(test_model, experiment_path,dct_param )


                artifact = wandb.Artifact('Plots', type='result')
                artifact.add_dir(experiment_path)#, name='images'
                wandb_logger.experiment.log_artifact(artifact, name ='plots')
                # wandb_logger.experiment.log_artifact(artifact_or_path=fig, name='pca_plot')

                #TODO Bring back in

                # neptune_logger.experiment.log_image('MCEs',"./results/images/mce_epochs_"+str(max_epochs)+".png")
                # neptune_logger.experiment.log_artifact("./results/images/mce_epochs_"+str(max_epochs)+".png")

                print('Test done')

#%%
# plot_ae_img(batch_features,test_loader)
# ls_dct_test =[{'a': 5},{'b': 10}]
# ls_x=[]
# ls_y=[]
# for mce in ls_dct_test:
#     for key, val in mce.items():
#         ls_x.append(key)
#         ls_y.append(val)
#
# import seaborn as sns
# sns.barplot(x=ls_x, y=ls_y)

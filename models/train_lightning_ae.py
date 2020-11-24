# pip install pytorch-lightning
# pip install neptune-client
# %%
from __future__ import print_function
import wandb
from utils.hessian_penalty_pytorch import hessian_penalty
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress import ProgressBar
import torch, torch.nn as nn, torchvision, torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import ast
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from collections import defaultdict
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
import pytorch_lightning as pl
# import utils.plot_utils as utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import manifold, decomposition
import pickle
import wandb
from scipy.stats import entropy
import time
import random
import os
from utils import disentangle_utils, training_utils, plot_utils, data_utils, utils, metric_utils, settings

# ToDo EDA:
# - Long Tail graphics
# - Remove user who had less than a threshold of seen items
# - Create Markdown with EDA results

# ToDo input_params:
# Parameter that should be tweakable by invoking the routine:
# - epochs
# - learning_rate
# - batch_size
# - simplified_rating
# - hidden_layer number
# - Algorithm: VAE, AE or SVD

# ToDo metrics:
# Add https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093

# ToDo training:
# Add test loss


seed = 42
torch.manual_seed(seed)


##This method creates a user-item matrix by transforming the seen items to 1 and adding unseen items as 0 if simplified_rating is set to True
##If set to False, the actual rating is taken
##Shape: (n_user, n_items)

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
    def __init__(self, conf: dict, *args, **kwargs):
        super().__init__()

        # self.kwargs = kwargs
        self.save_hyperparameters(conf)
        self.ls_predicted_movies = []
        self.is_hessian_penalty_activated = self.hparams["is_hessian_penalty_activated"]
        self.expanded_user_item = self.hparams["expanded_user_item"]
        self.used_data = self.hparams["used_data"]
        self.generative_factors = self.hparams["generative_factors"]
        self.mixup = self.hparams["mixup"]
        self.np_synthetic_data = self.hparams["synthetic_data"]
        self.ls_syn_y = self.hparams["syn_y"]
        self.experiment_path_train = conf["experiment_path"]
        self.experiment_path_test = self.experiment_path_train
        self.beta = self.hparams["beta"]
        self.avg_mce = 0.0
        self.train_dataset = None
        self.test_dataset = None
        self.test_size = self.hparams["test_size"]
        self.no_latent_factors = self.hparams["latent_dim"]
        self.max_unique_movies = 0
        self.unique_movies = 0
        self.np_user_item = None
        self.small_dataset = self.hparams["small_dataset"]
        self.simplified_rating = self.hparams["simplified_rating"]
        self.max_epochs = self.hparams["max_epochs"]
        self.dct_index2itemId = None
        self.test_y_bin = None
        self.df_movies_z_combined = None

        if (self.np_synthetic_data is None):
            self.load_dataset()  # additionaly assigns self.unique_movies and self.np_user_item
            self.df_movies = pd.read_csv('../data/generated/df_movies_cleaned3.csv')
            self.dct_attribute_distribution = utils.load_json_as_dict(
                'attribute_distribution.json')  # load relative frequency distributioon from dictionary (pickle it)

        else:
            self.train_dataset, self.test_dataset = train_test_split(self.np_synthetic_data, test_size=self.test_size,
                                                                     random_state=42)
            self.train_y, self.test_y = train_test_split(self.ls_syn_y, test_size=self.test_size, random_state=42)
            self.test_y_bin = np.asarray(pd.get_dummies(pd.DataFrame(data=self.test_y)))
            self.unique_movies = self.np_synthetic_data.shape[1]
            self.df_movies = pd.read_csv('../data/generated/syn.csv')
            self.dct_attribute_distribution = utils.load_json_as_dict(
                'syn_attribute_distribution.json')  # load relative frequency distributioon from dictionary (pickle it)

        # nn.Linear layer creates a linear function (θx + b), with its parameters initialized
        self.input_dimension = int(self.unique_movies * math.pow(4,
                                                                 self.generative_factors)) if self.expanded_user_item == True else self.unique_movies



        self.fc11 = nn.Linear(in_features=self.input_dimension, out_features=self.no_latent_factors)  # input

        self.fc12 = nn.Linear(in_features=40, out_features=self.no_latent_factors)  # input
        # self.fc13 = nn.Linear(in_features=1000, out_features=600)  # input
        self.encoder = nn.Sequential(self.fc11  ,# nn.ReLU(),
                                     #self.fc12#, nn.LeakyReLU()
                                     # self.fc13, nn.LeakyReLU()
                                     )

        self.fc31 = nn.Linear(in_features=self.no_latent_factors, out_features=40)
        # self.fc32 = nn.Linear(in_features=600, out_features=1000)
        # self.fc33 = nn.Linear(in_features=1000, out_features=1200)
        self.fc34 = nn.Linear(in_features=self.no_latent_factors, out_features=self.input_dimension)
        self.decoder = nn.Sequential(#self.fc31, nn.ReLU(),
                                     # self.fc32, nn.LeakyReLU(),
                                     # self.fc33, nn.ReLU(),
                                     self.fc34)

        self.KLD = None
        self.ls_kld = []
        self.dis_KLD = None

        self.z = None
        self.kld_matrix = np.empty((0, self.no_latent_factors))
        self.np_z_test = np.empty((0, self.no_latent_factors))  # self.test_dataset.shape[0]
        self.np_mu_test = np.empty((0, self.no_latent_factors))
        self.np_logvar_test = np.empty((0, self.no_latent_factors))

        self.np_z_train = np.empty((0, self.no_latent_factors))  # self.test_dataset.shape[0]
        self.np_mu_train = np.empty((0, self.no_latent_factors))
        self.np_logvar_train = np.empty((0, self.no_latent_factors))

        # self.dct_attribute_distribution = None # load relative frequency distributioon from dictionary (pickle it)

        # self.df_links = pd.read_csv('../data/movielens/small/links.csv')
        self.sigmoid_annealing_threshold = self.hparams['sigmoid_annealing_threshold']
        self.mce_batch_train = None
        self.mce_batch_test = None

        self.z_mean_train = []
        self.z_min_train = []
        self.z_max_train = []

        # Initialize weights
        self.encoder.apply(self.weight_init)
        self.decoder.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # nn.init.orthogonal_(m.weight)
            m.bias.data.zero_()


    def encode(self, x):
        return F.relu(self.encoder(x))


    def decode(self, z):
        return torch.sigmoid(self.decoder(z))


    def forward(self, x, **kwargs):
        # Si

        if (kwargs):
            z = kwargs['z']
            mu = kwargs['mu']
            logvar = kwargs['logvar']
            p = None
            q = None
        else:
            self.z = self.encode(x)
        return self.decode(self.z)



    def load_dataset(self):
        if (self.small_dataset):
            print("Load small dataset of ratings.csv")
            df_ratings = pd.read_csv("../data/movielens/small/ratings.csv")

        else:
            print("Load large dataset of ratings.csv")
            df_ratings = pd.read_csv("../data/movielens/large/ratings.csv")

        print('Shape of dataset:{}'.format(df_ratings.shape))
        self.np_user_item, self.unique_movies, self.max_unique_movies, self.dct_index2itemId = data_utils.pivot_create_user_item_matrix(
            df_ratings, True)  # manual_create_user_item_matrix(df_ratings, simplified_rating=self.simplified_rating)
        # self.np_user_item, self.max_unique_movies = manual_create_user_item_matrix(df_ratings, simplified_rating=self.simplified_rating)
        self.train_dataset, self.test_dataset = train_test_split(self.np_user_item, test_size=self.test_size,
                                                                 random_state=42)

    def train_dataloader(self):
        # TODO Change shuffle to True, just for dev purpose switched on
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=100, shuffle=True, num_workers=0, pin_memory=True
        )
        return train_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=100, shuffle=False, num_workers=0
        )
        return test_loader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)  # , weight_decay = 0.00001
        # criterion = nn.Binar()#MSELoss()  # mean-squared error loss
        # scheduler = StepLR(optimizer, step_size=1)
        return optimizer  # , scheduler

    def collect_z_values(self, ts_mu_chunk, ts_logvar_chunk):  # , ls_y
        start = time.time()
        ls_grad_z = self.compute_z(ts_mu_chunk, ts_logvar_chunk)
        self.np_z_train = np.append(self.np_z_train, np.asarray(ls_grad_z.tolist()),
                                    axis=0)  # TODO Describe in thesis that I get back a grad object instead of a pure tensor as it is in the test method since we are in the training method.

        # print('Shape np_z_train: {}'.format(self.np_z_train.shape))

        z_mean = self.np_z_train.mean(axis=0)
        z_min = self.np_z_train.min(axis=0)
        z_max = self.np_z_train.max(axis=0)

        if (len(self.z_mean_train) == 0):
            self.z_mean_train = z_mean
            self.z_min_train = z_min
            self.z_max_train = z_max

        else:
            self.z_mean_train = (z_mean + self.z_mean_train) / 2
            self.z_max_train = np.amax(np.vstack((self.z_max_train, z_max)),
                                       axis=0)  # Stack old and new together and find the max
            self.z_min_train = np.amin(np.vstack((self.z_min_train, z_min)), axis=0)
            # if (z_min < self.z_min_train):
            #     self.z_min_train = z_min
            #
            # if (z_max > self.z_max_train):
            #     self.z_max_train = z_max

        # print('collect_z_values in seconds: {}'.format(time.time() - start))



    def training_step(self, batch, batch_idx):
        mce_minibatch = None

        batch_len = batch.shape[0]
        ts_batch_user_features = batch  # .view(-1, self.input_dimension)

        # ts_batch_user_features = ts_batch_user_features * random.uniform(0.4,0.9)
        recon_batch = self.forward(ts_batch_user_features)  # sample data
        if (np.isnan(np.sum(recon_batch.detach().numpy()))):
            print('s')

        batch_mse = self.loss_function(recon_batch,
                                                  ts_batch_user_features,
                                                  self.beta,
                                                  self.unique_movies)
        hp_loss = 0
        # normalizing reconstruction loss
        batch_mse = batch_mse / len(ts_batch_user_features)

        if (self.is_hessian_penalty_activated and self.current_epoch > int(3 / 4 * self.max_epochs - 1)):  #
            print('<---- Applying Hessian Penalty ---->')
            # np_z = self.compute_z(ts_mu_chunk, ts_logvar_chunk)
            # hp_loss = hessian_penalty(G=self.decode, z=np_z)
            # print('Hessian Penalty:{}'.format(hp_loss))
            # batch_loss = batch_mse + hp_loss + batch_kld
        if (self.used_data == 'ae'):
            batch_loss = batch_mse

        # Additional logs go into tensorboard_logs
        tensorboard_logs = {'train_loss': batch_loss,
                            'MSE-Train': batch_mse,
                            }  #
        return {'loss': batch_loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        print("Saving MCE before KLD is applied...")
        return {}

        # def validation_step(self, batch, batch_idx):
        #     return 0

    def test_step(self, batch, batch_idx):
        print('test step')

        batch_mce = 0
        test_loss = 0

        # self.eval()
        ts_batch_user_features = batch.view(-1, self.input_dimension)


        recon_batch = self.forward(ts_batch_user_features)

        self.ls_predicted_movies.extend((-recon_batch).argsort()[:, 0].tolist())
        ls_z = self.encode(ts_batch_user_features).tolist()

        self.np_z_test = np.append(self.np_z_test, np.asarray(ls_z),
                                   axis=0)  # TODO get rid of np_z_chunk and use np.asarray(mu_chunk)

        batch_rmse_w_zeros, batch_mse_w_zeros, batch_rmse, batch_mse = self.calculate_batch_metrics(
            recon_batch=recon_batch, ts_batch_user_features=ts_batch_user_features)
        batch_mse = self.loss_function(recon_batch,
                                            ts_batch_user_features,
                                            self.beta,
                                            self.unique_movies)
        # normalizing reconstruction loss
        batch_mse = batch_mse / len(ts_batch_user_features)
        batch_loss = batch_mse


        # to be rermoved mean_mce = { for single_mce in batch_mce}
        loss = batch_loss.item() / len(ts_batch_user_features)

        # bce = batch_bce/len(ts_batch_user_features)
        tensorboard_logs = {
                            'MSE-test': batch_mse}

        return {'test_loss': loss,
                'rmse': batch_rmse,
                'mse': batch_mse,
                'rmse_w_zeros': batch_rmse_w_zeros,
                'mse_w_zeros': batch_mse_w_zeros,
                'log': tensorboard_logs,
                'MSE-Test': batch_mse
                }

        # test_loss /= len(test_loader.dataset)
        # print('====> Test set loss: {:.4f}'.format(test_loss))

    def test_epoch_end(self, outputs):
        # avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()


        avg_loss = np.array([x['test_loss'] for x in outputs]).mean()
        mse_test = np.array([x['MSE-Test'] for x in outputs])

        avg_rmse = np.array([x['rmse'] for x in outputs]).mean()
        avg_rmse_w_zeros = np.array([x['rmse_w_zeros'] for x in outputs]).mean()
        avg_mse = np.array([x['mse'] for x in outputs]).mean()
        avg_mse_w_zeros = np.array([x['mse_w_zeros'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss, 'MSE-Test': mse_test}

        wandb_logger.log_metrics({'rmse': avg_rmse,
                                  'rmse_w_zeros': avg_rmse_w_zeros,
                                  'mse': avg_mse,
                                  'mse_w_zeros': avg_mse_w_zeros})  # , 'kld_matrix':self.kld_matrix

        return {'test_loss': avg_loss, 'log': tensorboard_logs, 'rmse': avg_rmse, 'MSE-Test': mse_test
                }  # , , 'mce':avg_mce



        # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, beta, unique_movies):
        try:
            # MSE = F.binary_cross_entropy(recon_x, x, reduction='sum')# MSE is bad for this
            MSE = F.mse_loss(x, recon_x, reduction='sum')  # MSE is bad for this
            self.KLD = [0]
            if (np.isnan(np.sum(MSE.detach().numpy()))):
                print('s')
        except RuntimeError as e:
            print('fo', e)

        return MSE

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
            np_yhat_wo_zeros = np_yhat[np.nonzero(np_y)]  # This must be np_y

            rmse, mse = metric_utils.calculate_metrics(np_y, np_yhat)
            batch_mse += mse
            batch_rmse += rmse

            if (len(np_yhat_wo_zeros) > 0):
                rmse_wo_zeros, mse_wo_zeros = metric_utils.calculate_metrics(np_y_wo_zeros, np_yhat_wo_zeros)
                batch_rmse_wo_zeros += rmse_wo_zeros
                batch_mse_wo_zeros += mse_wo_zeros

        # batch_rmse, batch_mse = utils.calculate_metrics(ts_batch_user_features,ls_yhat_user)
        avg_rmse = batch_rmse / ls_yhat_user.shape[0]
        avg_rmse_wo_zeros = batch_rmse_wo_zeros / ls_yhat_user.shape[0]

        avg_mse = batch_mse / ls_yhat_user.shape[0]
        avg_mse_wo_zeros = batch_mse_wo_zeros / ls_yhat_user.shape[0]
        return avg_rmse, avg_mse, avg_rmse_wo_zeros, avg_mse_wo_zeros

    def load_attributes_and_files(self, path):  # 'filename.pickle'
        with open(path, 'rb') as handle:
            dct_attributes = pickle.load(handle)
        self.np_z_train = dct_attributes['np_z_train']
        self.np_logvar_train = dct_attributes['np_logvar_train']
        self.np_mu_train = dct_attributes['np_mu_train']
        self.train_y = dct_attributes['train_y']
        self.test_y = dct_attributes['test_y']
        self.ls_kld = dct_attributes['ls_kld']

        # self.dct_attribute_distribution = utils.load_json_as_dict(
        #     'attribute_distribution.json')  # load relative frequency distributioon from dictionary (pickle it)

        self.z_max_train = dct_attributes['z_max_train']
        self.z_min_train = dct_attributes['z_min_train']
        self.z_mean_train = dct_attributes['z_mean_train']
        print('Attributes loaded')

    def save_attributes(self, path):
        dct_attributes = {'np_z_train': self.np_z_train,
                          'np_logvar_train': self.np_logvar_train,
                          'np_mu_train': self.np_mu_train,
                          'train_y': self.train_y,
                          'test_y': self.test_y,
                          'ls_kld': self.ls_kld,
                          'z_max_train': self.z_max_train,
                          'z_min_train': self.z_min_train,
                          'z_mean_train': self.z_mean_train}
        with open(path, 'wb') as handle:
            pickle.dump(dct_attributes, handle)
        print('Attributes saved')

    def my_eval(expression):
        try:
            return ast.literal_eval(str(expression))
        except SyntaxError:  # e.g. a ":" or "(", which is interpreted by eval as command
            return [expression]
        except ValueError:  # e.g. an entry is nan, in that case just return an empty string
            return ''

    def calculate_mean_of_ls_dict(ls_dict: list):
        dct_sum = defaultdict(float)

        for dict in ls_dict:
            for key, val in dict.items():
                dct_sum[key] += val
        np_mean_vals = np.array(list(dct_sum.values())) / len(ls_dict)
        dct_mean = list(zip(dct_sum.keys(), np_mean_vals))
        # print(dct_mean)
        return dct_mean

    def match_metadata(indezes, df_links, df_movies, synthetic, dct_index2itemId):
        # ls_indezes = y_hat.values.index
        # TODO Source read_csv out
        # df_links = pd.read_csv('../data/movielens/small/links.csv')
        # df_movies = pd.read_csv('../data/generated/df_movies_cleaned3.csv')

        # if(synthetic == False):
        ls_filter = ['languages', 'directors', 'writer', 'writers',
                     'countries', 'runtimes', 'aspect_ratio', 'color_info',
                     'sound_mix', 'plot_outline', 'title', 'animation_department',
                     'casting_department', 'music_department', 'plot',
                     'set_decorators', 'script_department',
                     # TODO Add the attributes below once it works
                     'cast_id', 'stars_id', 'producers', 'language_codes',
                     'composers', 'cumulative_worldwide_gross', 'costume_designers',
                     'kind', 'editors', 'country_codes', 'assistant_directors', 'cast']
        df_movies_curated = df_movies.copy().drop(ls_filter, axis=1)
        ls_ml_ids = [dct_index2itemId[matrix_index] for matrix_index in indezes]  # ml = MovieLens

        sr_imdb_ids = df_links[df_links["movieId"].isin(ls_ml_ids)]['imdbId']  # If I want to keep the
        imdb_ids = sr_imdb_ids.array

        # print('no of imdbIds:{}, no of indezes:{}'.format(len(imdb_ids), len(indezes)))
        # TODO Fill df_movies with MovieLensId or download large links.csv
        if (len(imdb_ids) < len(indezes)):
            print(
                'There were items recommended that have not been seen by any users in the dataset. Trained on 9725 movies but 193610 are available so df_links has only 9725 to pick from')
        assert len(imdb_ids) == len(indezes)
        # df_links.loc[df_links["movieId"] == indezes]
        df_w_metadata = df_movies_curated.loc[df_movies_curated['imdbid'].isin(imdb_ids)]

        return df_w_metadata

    def alter_z(ts_z, latent_factor_position, model, strategy):
        if (strategy == 'max'):
            ts_z[:, latent_factor_position] = model.z_max_train[
                latent_factor_position]  # 32x no_latent_factors, so by accesing [:,pos] I get the latent factor for the batch of 32 users
        elif (strategy == 'min'):
            raise NotImplementedError("Min Strategy needs to be implemented")
        elif (strategy == 'min_max'):
            try:
                z_max_range = model.z_max_train[latent_factor_position]  # TODO Evtl. take z_max_train here
                z_min_range = model.z_min_train[latent_factor_position]

                if (np.abs(z_max_range) > np.abs(z_min_range)):
                    ts_z[:, latent_factor_position] = model.z_max_train[latent_factor_position]
                else:
                    ts_z[:, latent_factor_position] = model.z_min_train[latent_factor_position]
            except IndexError:
                print('stop')
        return ts_z


def generate_distribution_df():
    dct_attribute_distribution = utils.compute_relative_frequency(
        pd.read_csv('../data/generated/syn.csv'))
    utils.save_dict_as_json(dct_attribute_distribution, 'syn_attribute_distribution.json')

def traverse(test_model, experiment_path, dct_param):
    ls_axes = []
    fig = plt.figure(figsize=(25, 10))
    st = fig.suptitle('This is a somewhat long figure title', fontsize=16, y=1.3)
    # plt.subplots_adjust(top=0.25)
    # f, axes = plt.subplots(1, 2)

    for i in range(test_model.no_latent_factors):
        ax = fig.add_subplot(2, 5, i + 1)  # 2 rows, 5 cols, index 1
        ax.set_title('Latent Factor:{}'.format(i))
        ls_axes.append(ax)  # , ax2, ax3, ax4, ax5, ax6,ax7,ax8,ax9,ax10)

    # We use ax parameter to tell seaborn which subplot to use for this plot
    for idx, ax in enumerate(ls_axes):
        ls_y = []
        ls_tensor_values = []

        for i in range(-35, 35, 2):
            z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ls_y.append(i / 10)
            z[idx] = i / 10
            ls_tensor_values.append(z)

        np_fo = np.asarray(ls_tensor_values)
        p = torch.tensor(np_fo).float()
        # p = torch.tensor([[0, 0, 0, 0, 0, 0, 0, i, 0, 0]]).float()
        np_samples = test_model.decode(p).detach().numpy()

        ax_heatmap = sns.heatmap(np_samples, ax=ax, yticklabels=ls_y)
    # plt.title('Traversing over Latent Space', y=1.5)
    # plt.suptitle('Traversing over Latent Space',y=1.5)
    fig.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)

    plt.show()

    # ls_tensor_values=[]
    # z = [0, 0, 0, 0, 0, 0, -1, 0, -1, 0]
    # ls_tensor_values.append(z)
    # np_fo = np.asarray(ls_tensor_values)
    # p = torch.tensor(np_fo).float()
    # # p = torch.tensor([[0, 0, 0, 0, 0, 0, 0, i, 0, 0]]).float()
    # np_samples = test_model.decode(p).detach().numpy()
    #
    # ax_heatmap = sns.heatmap(np_samples)
    # plt.show()
    plot_utils.create_heatmap(np_samples, 'traversal', 'Sample ID', 'Item ID',
                              'heatmap-samples', experiment_path, dct_param)

if __name__ == '__main__':
    torch.manual_seed(100)
    args = training_utils.create_training_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available
    settings.init()
    # %%
    train = True
    synthetic_data = True
    expanded_user_item = False
    mixup = False
    is_hessian_penalty_activated = False
    # continous_data = False
    # normalvariate = False
    ls_normalvariate = [False]
    ls_continous = [True]
    base_path = 'results/models/vae/'
    used_data = 'syn'
    full_test_routine = False
    noise = False
    no_generative_factors = 3
    # used_data ='vae'
    used_data = 'ae'

    ls_epochs = [51]  # -->7 #5,10,15,20,25,30,40,50,60,70,80,90,100,120,150,200,270,350,500
    # Note: Mit steigender Epoche wird das disentanglement verstärkt
    #
    ls_latent_factors = [10]
    beta_normalized = 10 / (20 * no_generative_factors)
    ls_betas = []  # disentangle_factors .0003

    for epoch in ls_epochs:
        for normalvariate in ls_normalvariate:
            for continous_data in ls_continous:
                for lf in ls_latent_factors:
                    if (len(ls_betas) == 0):
                        if (expanded_user_item):
                            beta_normalized = lf / (800)
                        else:
                            beta_normalized = lf / (
                                        20 * no_generative_factors)  # lf/input_size, e.g. 2/10000 = 0.0002
                        ls_betas.append(beta_normalized)
                    for beta in ls_betas:
                        train_tag = "train"
                        if (not train):
                            train_tag = "test"

                        print(
                            "Processing model with: {} epochs, {} latent factors, {} beta".format(epoch, lf, beta))
                        # exp_name = "{}_beta_{}_epochs_{}_lf_synt_{}_normal_{}_continous_{}_hessian_{}_noise_{}".format(beta, epoch, lf, synthetic_data, normalvariate, continous_data, is_hessian_penalty_activated, noise)
                        exp_name = "ae-{}_beta_{}_epochs_{}_lf_synt_{}_normal_{}_continous_{}_hessian_{}".format(beta,
                                                                                                              epoch,
                                                                                                              lf,
                                                                                                              synthetic_data,
                                                                                                              normalvariate,
                                                                                                              continous_data,
                                                                                                              is_hessian_penalty_activated)
                        wandb_name = exp_name + "_" + train_tag
                        model_name = exp_name + ".ckpt"
                        attribute_name = exp_name + "_attributes.pickle"
                        model_path = base_path + model_name
                        attribute_path = base_path + attribute_name

                        experiment_path = utils.create_experiment_directory()

                        model_params = training_utils.create_model_params(experiment_path, epoch, lf, beta,
                                                                          int(epoch / 100), expanded_user_item,
                                                                          mixup,
                                                                          no_generative_factors, epoch,
                                                                          is_hessian_penalty_activated, used_data)

                        args.max_epochs = epoch

                        wandb_logger = WandbLogger(project='recommender-xai', tags=['vae', train_tag],
                                                   name=wandb_name)
                        trainer = pl.Trainer.from_argparse_args(args,
                                                                # limit_test_batches=0.1,
                                                                # precision =16,
                                                                logger=wandb_logger,  # False
                                                                gradient_clip_val=0.5,
                                                                # accumulate_grad_batches=0,
                                                                gpus=0,
                                                                weights_summary='full',
                                                                checkpoint_callback=False,
                                                                callbacks=[ProgressBar(),
                                                                           EarlyStopping(monitor='train_loss')]
                                                                )

                        if (train):
                            print(
                                '<---------------------------------- VAE Training ---------------------------------->')
                            print("Running with the following configuration: \n{}".format(args))
                            if (synthetic_data):
                                model_params['synthetic_data'], model_params[
                                    'syn_y'] = data_utils.create_synthetic_data(no_generative_factors,
                                                                                experiment_path,
                                                                                expanded_user_item,
                                                                                continous_data,
                                                                                normalvariate,
                                                                                noise)
                                generate_distribution_df()

                            model = VAE(model_params)
                            wandb_logger.watch(model, log='gradients', log_freq=100)

                            # utils.print_nn_summary(model, size =200)

                            print('------ Start Training ------')
                            trainer.fit(model)
                            kld_matrix = model.KLD
                            # print('% altering has provided information gain:{}'.format(
                            #     int(settings.ig_m_hat_cnt) / (int(settings.ig_m_cnt) + int(settings.ig_m_hat_cnt))))
                            # model.dis_KLD
                            print('------ Saving model ------')
                            trainer.save_checkpoint(model_path)
                            model.save_attributes(attribute_path)

                        print('------ Load model -------')
                        test_model = VAE.load_from_checkpoint(
                            model_path)  # , load_saved_attributes=True, saved_attributes_path='attributes.pickle'
                        # test_model.test_size = model_params['test_size']
                        test_model.load_attributes_and_files(attribute_path)
                        test_model.experiment_path_test = experiment_path

                        # print("show np_z_train mean:{}, min:{}, max:{}".format(z_mean_train, z_min_train, z_max_train ))
                        print('------ Start Test ------')
                        start = time.time()
                        dct_param = {'epochs': epoch, 'lf': lf, 'beta': beta, 'normal': normalvariate,
                                     'continous': continous_data, 'hessian': is_hessian_penalty_activated,
                                     'noise': noise}
                        # plot_utils.plot_samples(test_model, experiment_path, dct_param)

                        # z = torch.randn(1, test_model.no_latent_factors)
                        #
                        #
                        # Here we create a figure instance, and two subplots
                        traverse(test_model, experiment_path, dct_param)

                        trainer.test(test_model)  # The test loop will not be used until you call.
                        print('Test time in seconds: {}'.format(time.time() - start))
                        # print('% altering has provided information gain:{}'.format( int(settings.ig_m_hat_cnt)/(int(settings.ig_m_cnt)+int(settings.ig_m_hat_cnt) )))
                        # print(results)

                        disentangle_utils.run_disentanglement_eval(test_model, experiment_path, dct_param)

                        plot_utils.plot_results(test_model,
                                                test_model.experiment_path_test,
                                                test_model.experiment_path_train,
                                                dct_param)

                        artifact = wandb.Artifact('Plots', type='result')
                        artifact.add_dir(experiment_path)  # , name='images'
                        wandb_logger.experiment.log_artifact(artifact)

                        working_directory = os.path.abspath(os.getcwd())
                        absolute_path = working_directory + "/" + experiment_path + "images/"
                        ls_path_images = [absolute_path + file_name for file_name in os.listdir(absolute_path)]
                        # wandb.log({"images": [wandb.Image(plt.imread(img_path)) for img_path in ls_path_images]})

                        dct_images = {
                            img_path.split(sep='_')[2].split(sep='/')[-1]: wandb.Image(plt.imread(img_path)) for
                            img_path in ls_path_images}
                        wandb.log(dct_images)

                        # wandb.log({"example_1": wandb.Image(...), "example_2",: wandb.Image(...)})

                        # TODO Bring back in

                        # neptune_logger.experiment.log_image('MCEs',"./results/images/mce_epochs_"+str(max_epochs)+".png")
                        # neptune_logger.experiment.log_artifact("./results/images/mce_epochs_"+str(max_epochs)+".png")
                        print('Test done')

    exit()

# %%
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

# import plotly.express as px
# df = px.data.iris()
# print(df.head())


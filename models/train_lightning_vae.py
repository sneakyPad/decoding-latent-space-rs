# pip install pytorch-lightning
# pip install neptune-client
#%%
from __future__ import print_function
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch, torch.nn as nn, torchvision, torch.optim as optim
from tqdm import tqdm
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
# import recmetrics
# from surprise import Reader, SVD, Dataset
# from surprise.model_selection import train_test_split
import ast
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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
import math
import pytorch_lightning as pl
import utils.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import manifold, decomposition
import pickle
import wandb
from scipy.stats import entropy
import time
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


ig_m_cnt = 0
ig_m_hat_cnt =0
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
        self.expanded_user_item = conf["expanded_user_item"]
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
        self.unique_movies =0
        self.np_user_item = None
        self.small_dataset = self.hparams["small_dataset"]
        self.simplified_rating = self.hparams["simplified_rating"]
        self.max_epochs = self.hparams["max_epochs"]

        if(self.np_synthetic_data is None):
            self.load_dataset() #additionaly assigns self.unique_movies and self.np_user_item
            self.df_movies = pd.read_csv('../data/generated/df_movies_cleaned3.csv')
            self.dct_attribute_distribution = utils.load_json_as_dict(
                'attribute_distribution.json')  # load relative frequency distributioon from dictionary (pickle it)

        else:
            self.train_dataset, self.test_dataset = train_test_split(self.np_synthetic_data, test_size=self.test_size, random_state=42)
            self.train_y, self.test_y = train_test_split(self.ls_syn_y, test_size=self.test_size, random_state=42)

            self.unique_movies = self.np_synthetic_data.shape[1]
            self.df_movies = pd.read_csv('../data/generated/syn.csv')
            self.dct_attribute_distribution = utils.load_json_as_dict(
                'syn_attribute_distribution.json')  # load relative frequency distributioon from dictionary (pickle it)

        #nn.Linear layer creates a linear function (θx + b), with its parameters initialized
        input_dimension = 40*5*4 if self.expanded_user_item == True else 40

        self.fc1 = nn.Linear(in_features=input_dimension, out_features=400) #input
        self.fc11 = nn.Linear(in_features=400, out_features=100) #input
        self.encoder = nn.Sequential(self.fc1, self.fc11)

        self.fc21 = nn.Linear(in_features=100, out_features=self.no_latent_factors) #encoder mean
        self.fc22 = nn.Linear(in_features=100, out_features=self.no_latent_factors) #encoder variance
        self.fc3 = nn.Linear(in_features=self.no_latent_factors, out_features=100) #hidden layer, z

        self.fc41 = nn.Linear(in_features=100, out_features=400)
        self.fc42 = nn.Linear(in_features=400, out_features=input_dimension)
        self.decoder = nn.Sequential(self.fc41, self.fc42)

        self.KLD = None
        self.ls_kld = []
        self.dis_KLD = None

        self.z = None
        self.kld_matrix = np.empty((0, self.no_latent_factors))
        self.np_z_test = np.empty((0, self.no_latent_factors))#self.test_dataset.shape[0]
        self.np_mu_test = np.empty((0, self.no_latent_factors))
        self.np_logvar_test = np.empty((0, self.no_latent_factors))

        self.np_z_train = np.empty((0, self.no_latent_factors))  # self.test_dataset.shape[0]
        self.np_mu_train = np.empty((0, self.no_latent_factors))
        self.np_logvar_train = np.empty((0, self.no_latent_factors))

        # self.dct_attribute_distribution = None # load relative frequency distributioon from dictionary (pickle it)

        self.df_links = pd.read_csv('../data/movielens/small/links.csv')
        self.sigmoid_annealing_threshold = self.hparams['sigmoid_annealing_threshold']
        self.mce_batch_train = None
        self.mce_batch_test = None

        self.z_mean_train = []
        self.z_min_train = []
        self.z_max_train = []

        # Initialize weights
        self.encoder.apply(self.weight_init)
        self.decoder.apply(self.weight_init)

        # self.ig_m_hat_cnt = 0
        # self.ig_m_cnt = 0
        # if (kwargs.get('load_saved_attributes') == True):
        #     dct_attributes = self.load_attributes(kwargs.get('saved_attributes_path'))
        #     print('attributes loaded')
        #
        #     self.np_z_train = dct_attributes['np_z_train']

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # nn.init.orthogonal_(m.weight)
            m.bias.data.zero_()

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.encoder(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
        return torch.sigmoid(self.decoder(h3))

    def compute_z(self, mu, logvar):

        z = self.reparameterize(mu, logvar)
        return z

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def forward(self, x, **kwargs):
        #Si

        if(kwargs):
            z = kwargs['z']
            mu = kwargs['mu']
            logvar = kwargs['logvar']
            p = None
            q = None
        else:
            # print(x.view(-1, self.unique_movies)[0])
            # print(x[0])
            mu, logvar = self.encode(x) #40960/512 (Batchsize) results in 512,80
            # z = self.compute_z(mu, logvar)
            p, q, z = self.sample(mu, logvar)
            self.z = z

        return self.decode(z), mu, logvar, p, q

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

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
            self.train_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True
        )
        return train_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=16, shuffle=False, num_workers=0
        )
        return test_loader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.MSELoss()  # mean-squared error loss
        # scheduler = StepLR(optimizer, step_size=1)
        return optimizer#, scheduler

    def collect_z_values(self, ts_mu_chunk, ts_logvar_chunk):#, ls_y
        start = time.time()
        ls_grad_z = self.compute_z(ts_mu_chunk, ts_logvar_chunk)
        self.np_z_train = np.append(self.np_z_train, np.asarray(ls_grad_z.tolist()),
                                    axis=0)  # TODO Describe in thesis that I get back a grad object instead of a pure tensor as it is in the test method since we are in the training method.
        self.np_mu_train = np.append(self.np_mu_train, np.asarray(ts_mu_chunk.tolist()), axis=0)
        self.np_logvar_train = np.append(self.np_logvar_train, np.asarray(ts_logvar_chunk.tolist()), axis=0)

        print('Shape np_z_train: {}'.format(self.np_z_train.shape))


        z_mean = self.np_z_train.mean(axis=0)
        z_min = self.np_z_train.min(axis=0)
        z_max = self.np_z_train.max(axis=0)

        if(len(self.z_mean_train) == 0):
            self.z_mean_train = z_mean
            self.z_min_train = z_min
            self.z_max_train = z_max

        else:
            self.z_mean_train = (z_mean + self.z_mean_train) / 2
            self.z_max_train = np.amax(np.vstack((self.z_max_train, z_max)), axis=0) #Stack old and new together and find the max
            self.z_min_train = np.amin(np.vstack((self.z_min_train, z_min)), axis=0)
            # if (z_min < self.z_min_train):
            #     self.z_min_train = z_min
            #
            # if (z_max > self.z_max_train):
            #     self.z_max_train = z_max


        print('collect_z_values in seconds: {}'.format(time.time() - start))

    def average_mce_batch(self, mce_batch, mce_mini_batch):
        if (mce_batch == None):
            mce_batch = mce_mini_batch
        else:
            for key_lf, mce_lf in mce_batch.items():
                for key, val in mce_lf.items():
                    new_val = mce_mini_batch[key_lf].get(key)
                    if(new_val):
                        mce_batch[key_lf][key] = (new_val + val)/2
        return mce_batch

    #taken from https://github.com/facebookresearch/mixup-cifar10
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def training_step(self, batch, batch_idx):
        mce_minibatch=None

        print('train step')
        batch_len = batch.shape[0]
        ts_batch_user_features = batch.view(-1, 40*5*4)
        mixed_x, y_a, y_b, lam = self.mixup_data(ts_batch_user_features, ts_batch_user_features, alpha=1.0, use_cuda=False)

        recon_batch, ts_mu_chunk, ts_logvar_chunk, p, q = self.forward(mixed_x)  # sample data

        # ls_preference = self.train_y[batch_idx * batch_len :(batch_idx + 1) * batch_len]


        if(self.current_epoch == self.max_epochs-1):
            print("Last round..")
            self.collect_z_values(ts_mu_chunk, ts_logvar_chunk)#, ls_preference

        if (self.current_epoch == self.sigmoid_annealing_threshold ):
            self.collect_z_values(ts_mu_chunk, ts_logvar_chunk)
            mce_minibatch = mce_batch(self, mixed_x, k=3)
            self.mce_batch_train = self.average_mce_batch(self.mce_batch_train, mce_minibatch)

        batch_mse, batch_kld = self.loss_function(recon_batch,
                                                  mixed_x, #ts_batch_user_features,
                                                  ts_mu_chunk,
                                                  ts_logvar_chunk,
                                                  self.beta,
                                                  self.unique_movies,
                                                  p,
                                                  q,
                                                  new_kld_function = True)
        batch_loss = batch_mse + batch_kld
        self.ls_kld.append(self.KLD.tolist())

        #Additional logs go into tensorboard_logs
        tensorboard_logs = {'train_loss': batch_loss,
                'KLD-Train': batch_kld,
                'MSE-Train': batch_mse} #
        return {'loss': batch_loss, 'log': tensorboard_logs}


    def training_epoch_end(self, outputs):
        print("Saving MCE before KLD is applied...")
        if(self.current_epoch == self.sigmoid_annealing_threshold ):
            utils.save_dict_as_json(self.mce_batch_train, 'mce_results_wo_kld.json', self.experiment_path_train)
        return {}

    # def validation_step(self, batch, batch_idx):
    #     return 0

    def test_step(self, batch, batch_idx):
        print('test step')

        batch_mce =0
        test_loss = 0

        # self.eval()
        ts_batch_user_features = batch.view(-1, 40*5*4)
        mixed_x, y_a, y_b, lam = self.mixup_data(ts_batch_user_features, ts_batch_user_features, alpha=1.0, use_cuda=False)

        recon_batch, ts_mu_chunk, ts_logvar_chunk, p, q = self(mixed_x)
        ls_z = self.compute_z(ts_mu_chunk, ts_logvar_chunk)

        self.np_z_test = np.append(self.np_z_test, np.asarray(ls_z), axis=0) #TODO get rid of np_z_chunk and use np.asarray(mu_chunk)
        self.np_mu_test = np.append(self.np_mu_test, np.asarray(ts_mu_chunk), axis =0)
        self.np_logvar_test = np.append(self.np_logvar_test, np.asarray(ts_logvar_chunk), axis =0)
        # self.np_z = np.vstack((self.np_z, np_z_chunk))

        batch_rmse_w_zeros, batch_mse_w_zeros, batch_rmse, batch_mse = self.calculate_batch_metrics(recon_batch=recon_batch, ts_batch_user_features =ts_batch_user_features)
        batch_mse, kld = self.loss_function(recon_batch,
                                            mixed_x, #ts_batch_user_features,
                                            ts_mu_chunk,
                                            ts_logvar_chunk,
                                            self.beta,
                                            self.unique_movies,
                                            p,
                                            q,
                                            new_kld_function=True)
        batch_loss = batch_mse + kld

        mce_minibatch = mce_batch(self, ts_batch_user_features, k=3)
        self.mce_batch_test = self.average_mce_batch(self.mce_batch_test, mce_minibatch)

        #to be rermoved mean_mce = { for single_mce in batch_mce}
        loss = batch_loss.item() / len(ts_batch_user_features)

        # bce = batch_bce/len(ts_batch_user_features)
        tensorboard_logs = {'KLD-Test': kld,
                            'MSE-test': batch_mse}

        return {'test_loss': loss,
                'rmse': batch_rmse,
                'mse': batch_mse,
                'rmse_w_zeros': batch_rmse_w_zeros,
                'mse_w_zeros': batch_mse_w_zeros,
                'log':tensorboard_logs,
                'KLD-Test': kld,
                            'MSE-Test': batch_mse
                }

        # test_loss /= len(test_loader.dataset)
        # print('====> Test set loss: {:.4f}'.format(test_loss))

    def test_epoch_end(self, outputs):
        # avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_loss = np.array([x['test_loss'] for x in outputs]).mean()
        mse_test = np.array([x['MSE-Test'] for x in outputs])
        kld_test =np.array([x['KLD-Test'] for x in outputs])
        # ls_mce = {x['mce'] for x in outputs}
        # utils.save_dict_as_json(outputs[0]['mce'], 'mce_results.json', self.experiment_path) #TODO This is wrong as only one batch is processed, should be all
        utils.save_dict_as_json(self.mce_batch_test, 'mce_results.json', self.experiment_path_test)
        # avg_mce = dict(calculate_mean_of_ls_dict(ls_mce))

        avg_rmse = np.array([x['rmse'] for x in outputs]).mean()
        avg_rmse_w_zeros = np.array([x['rmse_w_zeros'] for x in outputs]).mean()
        avg_mse = np.array([x['mse'] for x in outputs]).mean()
        avg_mse_w_zeros = np.array([x['mse_w_zeros'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss, 'MSE-Test':mse_test,'KLD-Test': kld_test }
        assert len(mse_test)==len(kld_test)
        for i in range(0, len(mse_test)):
            wandb_logger.log_metrics({'MSE-Test': mse_test[i],'KLD-Test': kld_test[i]} )

        wandb_logger.log_metrics({'rmse': avg_rmse,
                                  'rmse_w_zeros':avg_rmse_w_zeros,
                                  'mse': avg_mse,
                                  'mse_w_zeros': avg_mse_w_zeros})#, 'kld_matrix':self.kld_matrix

        return {'test_loss': avg_loss, 'log': tensorboard_logs, 'rmse': avg_rmse, 'MSE-Test':mse_test,'KLD-test': kld_test }#, , 'mce':avg_mce

    def sigmoid_annealing(self, beta, epoch):

        stretch_factor = 0.5
        if(epoch < self.sigmoid_annealing_threshold):
            return 0
        else:
            kld_weight = beta/(1+ math.exp(-epoch * stretch_factor + self.sigmoid_annealing_threshold)) #epoch_threshold moves e function along the x-axis
            return kld_weight

    def kl_divergence(self,p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))


    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()

    def new_kld_func(self, p, q):
        log_qz = q.log_prob(self.z)
        log_pz = p.log_prob(self.z)

        kl = log_qz - log_pz

        return kl

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, beta, unique_movies, p, q, new_kld_function=False):
        from scipy.stats import norm
        zero_mask = generate_mask(x, recon_x, user_based_items_filter=True)
        one_mask = ~zero_mask
        # x = x[one_mask]
        # recon_x = recon_x[one_mask]
        # MSE = F.binary_cross_entropy(recon_x, x.view(-1, unique_movies),reduction='sum')  # TODO: Is that correct? binary cross entropy - (Encoder)
        MSE = F.mse_loss(x, recon_x)#/x.shape[1]

        if(new_kld_function):
            kl = self.new_kld_func(p,q)
            self.kld_matrix = np.append(self.kld_matrix, np.asarray(kl.tolist()), axis=0)
            kld_mean = kl.mean()
        else:
            # see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114

            # p = norm.pdf(recon_x[:,0].tolist(), mu[:,0].tolist(), logvar[:,0].tolist())
            # q = norm.pdf(x[:,0].tolist(), 0, 1)
            # kl_medium = self.kl_divergence(p,q)

            kld_latent_factors = torch.exp(logvar) + mu ** 2 - 1. - logvar
            kld_mean = -0.5 * torch.mean(torch.sum(-kld_latent_factors, dim=1)) #0: sum over latent factors (columns), 1: sum over sample (rows)
            self.kld_matrix = np.append(self.kld_matrix, np.asarray(kld_latent_factors.tolist()), axis=0)

        if(self.training):
            kld_weight = self.sigmoid_annealing(beta,self.current_epoch)
        else:
            kld_weight = beta

        self.KLD = kld_mean * kld_weight

        # CE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        # KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return MSE,  self.KLD


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
            np_yhat_wo_zeros = np_yhat[np.nonzero(np_y)] #This must be np_y

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

    def load_attributes_and_files(self, path): #'filename.pickle'
        with open(path, 'rb') as handle:
            dct_attributes = pickle.load(handle)
        self.np_z_train = dct_attributes['np_z_train']
        self.train_y = dct_attributes['train_y']
        self.test_y = dct_attributes['test_y']
        self.ls_kld = dct_attributes['ls_kld']

        # self.dct_attribute_distribution = utils.load_json_as_dict(
        #     'attribute_distribution.json')  # load relative frequency distributioon from dictionary (pickle it)

        self.z_max_train = dct_attributes['z_max_train']
        print('Attributes loaded')

    def save_attributes(self, path):
        dct_attributes = {'np_z_train':self.np_z_train,
                          'train_y': self.train_y,
                          'test_y':self.test_y,
                          'ls_kld':self.ls_kld,
                          'z_max_train': self.z_max_train}
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

def mce_relative_frequency(y_hat, y_hat_latent, dct_attribute_distribution):
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
                        if(value in ls_y_attribute_val): #if no change, assign highest error
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

def shannon_inf_score(m, m_hat):
    epsilon = 1e-10
    shannon_inf = - math.log(m_hat) + epsilon
    mce = 1 / shannon_inf * math.exp(m_hat - m)

    if (mce < 0):
        print('fo')
    if (mce > 15):
        mce = 15
    if (math.isnan(mce)):
        print('nan detected')
    return mce

def calculate_normalized_entropy(population):
    H=0
    H_n =0
    try:
        if(len(population) == 1): #only one attribute is present
            H = - (population[0] * math.log(population[0], 2))
            return H
        else:
            # for rf in population:
            #     H_n = - (rf * math.log(rf, 2)) /math.log(len(population), 2)


            H_scipy = entropy(population, base=2)
            H_n = H_scipy / math.log(len(population), 2)
            # print(H_scipy, H_n)
            # assert H_scipy == H_n
    except ValueError:
        print('f')


    return H_n

def information_gain(m, m_hat, dct_population):
    global ig_m_cnt
    global ig_m_hat_cnt
    ls_population_rf = [val for key, val in dct_population.items()]
    population_entropy = calculate_normalized_entropy(ls_population_rf)

    if(type(m) is not list):
        m = [m]
    if(type(m_hat) is not list):
        m_hat = [m_hat]

    m_entropy = calculate_normalized_entropy(m)
    m_hat_entropy = calculate_normalized_entropy(m_hat)

    ig_m = population_entropy - m_entropy
    ig_m_hat = population_entropy - m_hat_entropy

    if(ig_m_hat > ig_m): #This means it was more unlikely so we gain information. Goal is to get to 1
        ig_m_hat_cnt +=1
        # return ig_m_hat

    ig_m_cnt += 1
    return ig_m_hat - ig_m

def mce_information_gain(y_hat, y_hat_latent, dct_attribute_distribution):
    # dct_dist = pickle.load(movies_distribution)

    dct_mce = defaultdict(float)
    for idx_vector in range(y_hat.shape[0]):
        for attribute in y_hat:
            if(attribute not in ['Unnamed: 0', 'unnamed_0', 'plot_outline','id']):
                ls_y_attribute_val = my_eval(y_hat.iloc[idx_vector][attribute]) #e.g. Stars: ['Pitt', 'Damon', 'Jolie']
                ls_y_latent_attribute_val = my_eval(y_hat_latent.iloc[idx_vector][attribute]) #e.g Stars: ['Depp', 'Jolie']
                mean = 0
                cnt_same = 0
                mce=0
                m_hat = 0
                m = 0

                try:
                    #Two cases: Either cell contains multiple values, than it is a list
                    #or it contains a single but not in a list. In that case put it in a list
                    if(type(ls_y_latent_attribute_val) is not list):
                        ls_y_latent_attribute_val = [ls_y_latent_attribute_val]
                        ls_y_attribute_val =[ls_y_attribute_val]

                    if (len(ls_y_attribute_val) == 0):
                        break

                    ls_m_hat_rf=[]
                    #Go through elements of a cell
                    for value in ls_y_latent_attribute_val: #same as characteristic
                        # if(value in ls_y_attribute_val): #if no change, assign highest error
                            # mean += 1
                            # m_hat +=1

                            # ls_y_latent_attribute_val.pop(value)

                        # else:
                            # characteristic = y_hat.loc[idx_vector, attribute]
                            y_hat_latent_attribute_relative_frequency = dct_attribute_distribution[attribute]['relative'][str(value)]
                            m_hat += y_hat_latent_attribute_relative_frequency
                            ls_m_hat_rf.append(y_hat_latent_attribute_relative_frequency)
                            # print('\t Value: {}, Relative frequency:{}'.format(value, relative_frequency))
                    #if no values are presented in the current cell than assign highest error
                    if(len(ls_y_latent_attribute_val)==0):
                        # mce =15
                        #TODO sth else than just break, maybe mce = -1?
                        break
                    else:
                        #rf = relative frequency
                        dct_population = dct_attribute_distribution[attribute]['relative']

                        ls_y_hat_rf = [dct_population[str(val)] for val in ls_y_attribute_val]
                        m = np.asarray(ls_y_hat_rf).mean()
                        m_hat = m_hat/len(ls_y_latent_attribute_val)

                        mce = information_gain(ls_y_hat_rf, ls_m_hat_rf, dct_population)
                        # mce = shannon_inf_score(m, m_hat)

                    prev_mce = dct_mce.get(attribute)
                    if (prev_mce):
                        dct_mce[attribute] = (prev_mce + mce) / 2
                    else:
                        dct_mce[attribute] = mce
                except (KeyError, TypeError, ZeroDivisionError) as e:
                    print("Error Value:{}".format(value))

    return dct_mce



def mce_shannon_inf(y_hat, y_hat_latent, dct_attribute_distribution):
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
                m_hat = 0
                m = 0

                try:
                    #Two cases: Either cell contains multiple values, than it is a list
                    #or it contains a single but not in a list. In that case put it in a list
                    if(type(ls_y_latent_attribute_val) is not list):
                        ls_y_latent_attribute_val = [ls_y_latent_attribute_val]
                        ls_y_attribute_val =[ls_y_attribute_val]

                    if (len(ls_y_attribute_val) == 0):
                        break

                    #Go through elements of a cell
                    for value in ls_y_latent_attribute_val: #same as characteristic
                        if(value in ls_y_attribute_val): #if no change, assign highest error
                            # mean += 1
                            m_hat +=1
                            # ls_y_latent_attribute_val.pop(value)

                        else:
                            # characteristic = y_hat.loc[idx_vector, attribute]
                            y_hat_latent_attribute_relative_frequency = dct_attribute_distribution[attribute]['relative'][str(value)]
                            m_hat += y_hat_latent_attribute_relative_frequency
                            # print('\t Value: {}, Relative frequency:{}'.format(value, relative_frequency))
                    #if no values are presented in the current cell than assign highest error
                    if(len(ls_y_latent_attribute_val)==0):
                        mce =15
                    else:
                        #rf = relative frequency

                        ls_y_hat_rf = [dct_attribute_distribution[attribute]['relative'][str(val)] for val in ls_y_attribute_val]
                        m = np.asarray(ls_y_hat_rf).mean()
                        m_hat = m_hat/len(ls_y_latent_attribute_val)

                        mce = shannon_inf_score(m, m_hat)

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

def match_metadata(indezes, df_links, df_movies, synthetic):
    # ls_indezes = y_hat.values.index
    #TODO Source read_csv out
    # df_links = pd.read_csv('../data/movielens/small/links.csv')
    # df_movies = pd.read_csv('../data/generated/df_movies_cleaned3.csv')

    global dct_index2itemId
    # if(synthetic == False):
    ls_filter = ['languages','directors','writer', 'writers',
                 'countries','runtimes', 'aspect_ratio', 'color_info',
                 'sound_mix', 'plot_outline', 'title', 'animation_department',
                 'casting_department', 'music_department','plot',
                 'set_decorators', 'script_department',
                 #TODO Add the attributes below once it works
                 'cast_id', 'stars_id', 'producers', 'language_codes',
                 'composers', 'cumulative_worldwide_gross','costume_designers',
                 'kind', 'editors','country_codes', 'assistant_directors', 'cast']
    df_movies_curated = df_movies.copy().drop(ls_filter,axis=1)
    ls_ml_ids = [dct_index2itemId[matrix_index] for matrix_index in indezes] #ml = MovieLens


    sr_imdb_ids = df_links[df_links["movieId"].isin(ls_ml_ids)]['imdbId'] #If I want to keep the
    imdb_ids = sr_imdb_ids.array

    # print('no of imdbIds:{}, no of indezes:{}'.format(len(imdb_ids), len(indezes)))
    #TODO Fill df_movies with MovieLensId or download large links.csv
    if(len(imdb_ids) < len(indezes)):
        print('There were items recommended that have not been seen by any users in the dataset. Trained on 9725 movies but 193610 are available so df_links has only 9725 to pick from')
    assert len(imdb_ids) == len(indezes)
    #df_links.loc[df_links["movieId"] == indezes]
    df_w_metadata = df_movies_curated.loc[df_movies_curated['imdbid'].isin(imdb_ids)]

    return df_w_metadata


def alter_z(ts_z, latent_factor_position, model):
    ts_z[:, latent_factor_position] = model.z_max_train[latent_factor_position] #32x no_latent_factors, so by accesing [:,pos] I get the latent factor for the batch of 32 users
    return ts_z

#MCE is calculated for each category
def mce_batch(model, ts_batch_features, k=0):
    dct_mce_mean = defaultdict()
    # hold n neurons of hidden layer
    # change 1 neuron
    ls_y_hat, mu, logvar, p, q = model(ts_batch_features)
    z = model.z
    # dct_attribute_distribution = utils.load_json_as_dict('attribute_distribution.json') #    load relative frequency distributioon from dictionary (pickle it)

    for latent_factor_position in range(model.no_latent_factors):
        print("Calculate MCEs for position: {} in vector z".format(latent_factor_position))
        ts_altered_z = alter_z(z, latent_factor_position, model)

        ls_y_hat_latent_changed, mu, logvar, p, q = model(ts_batch_features, z=ts_altered_z, mu=mu,logvar=logvar)

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

            y_hat_k_highest = (-y_hat).argsort()[:1] #Alternative: (-y_hat).sort().indices[:no_of_seen_items]
            y_hat_latent_k_highest = (-y_hat_latent).argsort()[:k] #Alternative: (-y_hat).sort().indices[:no_of_seen_items]

            # for 
            if(model.np_synthetic_data is None):
                # synthetic = False
                y_hat_w_metadata = match_metadata(y_hat_k_highest.tolist(), model.df_links, model.df_movies)
                y_hat_latent_w_metadata = match_metadata(y_hat_latent_k_highest.tolist(), model.df_links, model.df_movies)
            else:
                y_hat_w_metadata = model.df_movies.loc[model.df_movies['id'].isin(y_hat_k_highest.tolist())]
                y_hat_latent_w_metadata = model.df_movies.loc[model.df_movies['id'].isin(y_hat_latent_k_highest.tolist())]

            # single_mce = mce_relative_frequency(y_hat_w_metadata, y_hat_latent_w_metadata, model.dct_attribute_distribution) #mce for n columns
            # single_mce = mce_shannon_inf(y_hat_w_metadata, y_hat_latent_w_metadata, model.dct_attribute_distribution) #mce for n columns
            # single_mce = mce_information_gain(y_hat_w_metadata, y_hat_latent_w_metadata, model.dct_attribute_distribution) #mce for n columns
            # ls_dct_mce.append(single_mce)

            # print(single_mce)
        dct_mce_mean[latent_factor_position] = dict(calculate_mean_of_ls_dict(ls_dct_mce))
    return dct_mce_mean


def generate_distribution_df():
    dct_attribute_distribution = utils.compute_relative_frequency(
        pd.read_csv('../data/generated/syn.csv'))
    utils.save_dict_as_json(dct_attribute_distribution, 'syn_attribute_distribution.json')

if __name__ == '__main__':

    train_dataset = None
    test_dataset = None
    max_epochs = 20

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

    model_params = {"simplified_rating": True,
                    "small_dataset": True,
                    "test_size": 0.15,#TODO Change test size to 0.33
                    "latent_dim": 3,
                    "beta":1,
                    "sigmoid_annealing_threshold": 0,
                    "max_epochs": max_epochs}
    # model_params.update(args.__dict__)
    # print(**model_params)

    merged_params = (lambda first_dict, second_dict: {**first_dict, **second_dict})(args.__dict__, model_params)
    # print(merged_params)

    #%%
    train = True
    synthetic_data = True
    expanded_user_item = False
    hessian_penalty = False
    base_path = 'results/models/vae/'

    ls_epochs = [700]
    ls_latent_factors = [2]
    ls_betas = [] #disentangle_factors .0003
    no_generative_factors = 2
    # ls_latent_factors = [4]
    # ls_betas = [0.001] #disentangle_factors
    #TODO
    # beta_normalized = lf/input_size, e.g. 2/10000 = 0.0002
    for epoch in ls_epochs:
        for lf in ls_latent_factors:
            if(len(ls_betas)==0):
                if(expanded_user_item):
                    beta_normalized = lf/(800)
                else:
                    beta_normalized = lf / (20 * no_generative_factors)
                ls_betas.append(beta_normalized)
            for beta in ls_betas:
                train_tag = "train"
                if(not train):
                    train_tag = "test"

                print("Processing model with: {} epochs, {} latent factors, {} beta".format(epoch, lf, beta))
                exp_name = "{}_beta_{}_epochs_{}_lf_synt_{}".format(beta, epoch, lf, synthetic_data)
                wandb_name = exp_name + "_" + train_tag
                model_name = exp_name + ".ckpt"
                attribute_name = exp_name + "_attributes.pickle"
                model_path = base_path + model_name
                attribute_path = base_path + attribute_name

                experiment_path = utils.create_experiment_directory()


                model_params['experiment_path'] = experiment_path
                model_params['max_epochs'] = epoch
                model_params['latent_dim'] = lf
                model_params['beta'] = beta
                model_params['synthetic_data'] = None
                model_params['sigmoid_annealing_threshold'] = int(epoch/6)
                model_params['expanded_user_item'] = expanded_user_item

                args.max_epochs = epoch

                wandb_logger = WandbLogger(project='recommender-xai', tags=['vae', train_tag], name=wandb_name)
                trainer = pl.Trainer.from_argparse_args(args,
                                                        logger=wandb_logger, #False
                                                        gpus=0,
                                                        weights_summary='full',
                                                        checkpoint_callback = False,
                                                        callbacks = [EarlyStopping(monitor='train_loss')]
                )


                if(train):
                    print('<---------------------------------- VAE Training ---------------------------------->')
                    print("Running with the following configuration: \n{}".format(args))
                    if (synthetic_data):
                        model_params['synthetic_data'], model_params['syn_y'] = utils.create_synthetic_data(no_generative_factors, experiment_path, expanded_user_item)
                        generate_distribution_df()

                    model = (model_params)
                    # wandb_logger.watch(model, log='gradients', log_freq=100)

                    # utils.print_nn_summary(model, size =200)

                    print('------ Start Training ------')
                    trainer.fit(model)

                    kld_matrix = model.KLD
                    # print('% altering has provided information gain:{}'.format(
                    #     int(ig_m_hat_cnt) / (int(ig_m_cnt) + int(ig_m_hat_cnt))))

                    # model.dis_KLD
                    print('------ Saving model ------')
                    trainer.save_checkpoint(model_path)
                    model.save_attributes(attribute_path)

                print('------ Load model -------')
                test_model = VAE.load_from_checkpoint(model_path)#, load_saved_attributes=True, saved_attributes_path='attributes.pickle'
                # test_model.test_size = model_params['test_size']
                test_model.load_attributes_and_files(attribute_path)
                test_model.experiment_path_test = experiment_path

                # print("show np_z_train mean:{}, min:{}, max:{}".format(z_mean_train, z_min_train, z_max_train ))
                print('------ Start Test ------')
                start = time.time()
                ig_m_hat_cnt = 0
                ig_m_cnt = 0
                trainer.test(test_model) #The test loop will not be used until you call.
                print('Test time in seconds: {}'.format(time.time() - start))
                # print('% altering has provided information gain:{}'.format( int(ig_m_hat_cnt)/(int(ig_m_cnt)+int(ig_m_hat_cnt) )))
                # print(results)

                dct_param ={'epochs':epoch, 'lf':lf,'beta':beta}
                utils.plot_results(test_model,
                                   test_model.experiment_path_test,
                                   test_model.experiment_path_train,
                                   dct_param )

                artifact = wandb.Artifact('Plots', type='result')
                artifact.add_dir(experiment_path)#, name='images'
                wandb_logger.experiment.log_artifact(artifact)


                working_directory = os.path.abspath(os.getcwd())
                absolute_path = working_directory + "/" + experiment_path + "images/"
                ls_path_images = [absolute_path + file_name for file_name in os.listdir(absolute_path)]
                # wandb.log({"images": [wandb.Image(plt.imread(img_path)) for img_path in ls_path_images]})

                dct_images = {img_path.split(sep='_')[2].split(sep='/')[-1]: wandb.Image(plt.imread(img_path)) for img_path in ls_path_images}
                wandb.log(dct_images)



                # wandb.log({"example_1": wandb.Image(...), "example_2",: wandb.Image(...)})


                #TODO Bring back in

                # neptune_logger.experiment.log_image('MCEs',"./results/images/mce_epochs_"+str(max_epochs)+".png")
                # neptune_logger.experiment.log_artifact("./results/images/mce_epochs_"+str(max_epochs)+".png")
                print('Test done')

    exit()

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

# import plotly.express as px
# df = px.data.iris()
# print(df.head())


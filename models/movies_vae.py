# pip install pytorch-lightning
from __future__ import print_function
from utils.hessian_penalty.hessian_penalty_pytorch import hessian_penalty
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress import ProgressBar
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from collections import defaultdict
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import math
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import wandb
import time
import os
from utils import training_utils, plot_utils, data_utils, utils, metric_utils, settings, latent_space_utils, \
    disentangle_utils

#ToDo EDA:
# - Long Tail graphic
# - Remove user who had less than a certain threshold of seen items

#ToDo input_params:
# Parameter that should be adjustable by invoking the routine:
# - epochs
# - learning_rate
# - batch_size
# - simplified_rating

#ToDo metrics:
# Add https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093

seed = 42
torch.manual_seed(seed)

class VAE(pl.LightningModule):
    def __init__(self, conf:dict, *args, **kwargs):
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
        self.dct_index2itemId = None
        self.test_y_bin = None
        self.df_movies_z_combined =None

        if(self.np_synthetic_data is None):
            self.load_dataset() #additionaly assigns self.unique_movies and self.np_user_item
            self.df_movies = pd.read_csv('../data/generated/df_movies_cleaned3.csv')
            self.df_links = pd.read_csv('../data/movielens/small/links.csv')
            self.dct_attribute_distribution = utils.load_json_as_dict(
                'attribute_distribution.json')  # load relative frequency distributioon from dictionary (pickle it)

        else:
            self.ls_syn_y = self.hparams["syn_y"]
            self.df_links = pd.read_csv('../data/movielens/small/links.csv')
            self.test_size = 0.05
            self.train_dataset, self.test_dataset = train_test_split(self.np_synthetic_data, test_size=self.test_size, random_state=42)
            self.train_y, self.test_y = train_test_split(self.ls_syn_y, test_size=self.test_size, random_state=42)
            self.test_y_bin = np.asarray(pd.get_dummies(pd.DataFrame(data=self.test_y)))
            self.unique_movies = self.np_synthetic_data.shape[1]
            self.df_movies = pd.read_csv('../data/generated/syn.csv')
            self.dct_attribute_distribution = utils.load_json_as_dict(
                'syn_attribute_distribution.json')  # load relative frequency distributioon from dictionary (pickle it)

        #nn.Linear layer creates a linear function (θx + b), with its parameters initialized
        self.input_dimension = int(self.unique_movies *math.pow(4, self.generative_factors)) if self.expanded_user_item == True else self.unique_movies

        self.fc11 = nn.Linear(in_features=self.input_dimension, out_features=200)  # input
        # self.fc12 = nn.Linear(in_features=600, out_features=300)  # input
        # self.fc13 = nn.Linear(in_features=1000, out_features=600)  # input
        self.encoder = nn.Sequential(self.fc11#, nn.LeakyReLU(),
                                     # self.fc12, nn.LeakyReLU()
                                     # self.fc13, nn.LeakyReLU()
                                     )

        self.fc21 = nn.Linear(in_features=200, out_features=self.no_latent_factors)  # encoder mean
        self.fc22 = nn.Linear(in_features=200, out_features=self.no_latent_factors)  # encoder variance

        self.fc31 = nn.Linear(in_features=self.no_latent_factors, out_features=200)
        # self.fc32 = nn.Linear(in_features=600, out_features=1000)
        # self.fc33 = nn.Linear(in_features=1000, out_features=1200)
        self.fc34 = nn.Linear(in_features=200, out_features=self.input_dimension)
        self.decoder = nn.Sequential(self.fc31, nn.LeakyReLU(),
                                     # self.fc32, nn.LeakyReLU(),
                                     # self.fc33, nn.ReLU(),
                                     self.fc34)

        self.KLD = None
        self.ls_kld = []
        self.ls_kld_test = []
        self.dis_KLD = None

        self.z = None
        self.kld_matrix_train = np.empty((0, self.no_latent_factors))
        self.kld_matrix_test = np.empty((0, self.no_latent_factors))
        self.np_z_test = np.empty((0, self.no_latent_factors))#self.test_dataset.shape[0]
        self.np_mu_test = np.empty((0, self.no_latent_factors))
        self.np_logvar_test = np.empty((0, self.no_latent_factors))

        self.np_z_train = np.empty((0, self.no_latent_factors))  # self.test_dataset.shape[0]
        self.np_mu_train = np.empty((0, self.no_latent_factors))
        self.np_logvar_train = np.empty((0, self.no_latent_factors))
        self.np_var_train = np.empty((0, self.no_latent_factors))
        self.np_var_test = np.empty((0, self.no_latent_factors))

        # self.dct_attribute_distribution = None # load relative frequency distributioon from dictionary (pickle it)

        self.sigmoid_annealing_threshold = self.hparams['sigmoid_annealing_threshold']
        self.mce_batch_train = None
        self.mce_batch_test = None

        self.z_mean_train = []
        self.z_min_train = []
        self.z_max_train = []

        # Initialize weights
        self.encoder.apply(training_utils.weight_init)
        self.decoder.apply(training_utils.weight_init)

        # if (kwargs.get('load_saved_attributes') == True):
        #     dct_attributes = self.load_attributes(kwargs.get('saved_attributes_path'))
        #     print('attributes loaded')
        #     self.np_z_train = dct_attributes['np_z_train']

    def encode(self, x):
        """Encode a batch of samples, and return posterior parameters for each point."""
        h1 = F.relu(self.encoder(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """Reparameterisation trick to sample z values.
        This is stochastic during training,  and returns the mode during evaluation."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reparameterize_wo_eps(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return mu + std

    def decode(self, z):
        """Decode a batch of latent variables"""
        return torch.sigmoid(self.decoder(z))

    def compute_z(self, mu, logvar):
        """Encode a batch of data points, x, into their z representations."""
        z = self.reparameterize(mu, logvar)
        return z

    def sample(self, mu, log_var):
        # if (self.training):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def forward(self, x, **kwargs):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""

        if(kwargs):
            z = kwargs['z']
            mu = kwargs['mu']
            logvar = kwargs['logvar']
            p = None
            q = None
        else:
            mu, logvar = self.encode(x) #40960/512 (Batchsize) results in 512,80

            z = self.reparameterize(mu,logvar)
            # z = self.compute_z(mu, logvar)
            self.z = z
            p, q, _z = self.sample(mu, logvar)

        return self.decode(z), mu, logvar, p, q

    def load_dataset(self):
        if (self.small_dataset):
            print("Load small dataset of ratings.csv")
            df_ratings = pd.read_csv("../data/movielens/small/ratings.csv")

        else:
            print("Load large dataset of ratings.csv")
            df_ratings = pd.read_csv("../data/movielens/large/ml-25m/ratings.csv")
            df_ratings = df_ratings.iloc[:300000]

        print('Shape of dataset:{}'.format(df_ratings.shape))
        self.np_user_item, self.unique_movies, self.max_unique_movies, self.dct_index2itemId = data_utils.pivot_create_user_item_matrix(df_ratings,False)#manual_create_user_item_matrix(df_ratings, simplified_rating=self.simplified_rating)
        # self.np_user_item, self.max_unique_movies = manual_create_user_item_matrix(df_ratings, simplified_rating=self.simplified_rating)
        self.train_dataset, self.test_dataset = train_test_split(self.np_user_item, test_size=self.test_size, random_state=42)

    def train_dataloader(self):
        #TODO Change shuffle to True, just for dev purpose switched on
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=100, shuffle=True, num_workers=0, pin_memory=True
        )
        return train_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=10, shuffle=False, num_workers=0
        )
        return test_loader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3) #, weight_decay = 0.00001
        # criterion = nn.Binar()#MSELoss()  # mean-squared error loss
        # scheduler = StepLR(optimizer, step_size=1)
        return optimizer#, scheduler

    def collect_z_values(self, ts_mu_chunk, ts_logvar_chunk):#, ls_y
        start = time.time()
        ls_grad_z = self.compute_z(ts_mu_chunk, ts_logvar_chunk)
        self.np_z_train = np.append(self.np_z_train, np.asarray(ls_grad_z.tolist()),
                                    axis=0)  # TODO Describe in thesis that I get back a grad object instead of a pure tensor as it is in the test method since we are in the training method.

        # print('Shape np_z_train: {}'.format(self.np_z_train.shape))


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


        # print('collect_z_values in seconds: {}'.format(time.time() - start))

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

    def training_step(self, batch, batch_idx):
        mce_minibatch=None

        # if(self.current_epoch > 0):

            # print('self.fc11: ', np.isnan(np.sum(self.fc11.weight.grad.detach().numpy())))

            # output = np.isnan(np.sum(self.fc11.weight.detach().numpy())) \
              #
            # if(output):
            #     print('ho')
            # print('train step')
        batch_len = batch.shape[0]
        ts_batch_user_features = batch#.view(-1, self.input_dimension)
        if(self.mixup):
            ts_batch_user_features, y_a, y_b, lam = self.mixup_data(ts_batch_user_features, ts_batch_user_features, alpha=1.0, use_cuda=False)

        if (self.current_epoch == self.sigmoid_annealing_threshold):

            print('stop')

        # ts_batch_user_features = ts_batch_user_features * random.uniform(0.4,0.9)
        recon_batch, ts_mu_chunk, ts_logvar_chunk, p, q = self.forward(ts_batch_user_features)  # sample data
        if(np.isnan(np.sum(recon_batch.detach().numpy()))):
            print('s')
        # ls_preference = self.train_y[batch_idx * batch_len :(batch_idx + 1) * batch_len]


        if(self.current_epoch == self.max_epochs-1):
            # print("Last round..")
            self.collect_z_values(ts_mu_chunk, ts_logvar_chunk)#, ls_preference

        if (self.current_epoch == self.sigmoid_annealing_threshold ):
            self.collect_z_values(ts_mu_chunk, ts_logvar_chunk)
            # mce_minibatch = metric_utils.mce_batch(self, ts_batch_user_features, self.dct_index2itemId, k=3)
            # self.mce_batch_train = self.average_mce_batch(self.mce_batch_train, mce_minibatch)

        batch_mse, batch_kld = self.loss_function(recon_batch,
                                                  ts_batch_user_features, #ts_batch_user_features,
                                                  ts_mu_chunk,
                                                  ts_logvar_chunk,
                                                  self.beta,
                                                  self.unique_movies,
                                                  p,
                                                  q,
                                                  new_kld_function = False)
        hp_loss =0
        #normalizing reconstruction loss
        batch_mse = batch_mse / len(ts_batch_user_features)

        if(self.is_hessian_penalty_activated and self.current_epoch > int(3/4*self.max_epochs-1)):#
            print('<---- Applying Hessian Penalty ---->')
            np_z = self.compute_z(ts_mu_chunk, ts_logvar_chunk)
            hp_loss = hessian_penalty(G=self.decode, z=np_z)
            print('Hessian Penalty:{}'.format(hp_loss))
            batch_loss = batch_mse + hp_loss + batch_kld
        else:
            batch_loss = batch_mse + batch_kld

        self.ls_kld.append(self.KLD.tolist())
        #Additional logs go into tensorboard_logs
        tensorboard_logs = {'train_loss': batch_loss,
                'KLD-Train': batch_kld,
                'MSE-Train': batch_mse,
                            } #
        return {'loss': batch_loss, 'log': tensorboard_logs,
                'var': np.exp(np.asarray(ts_logvar_chunk.tolist())).mean(axis=0),
                'logvar': np.asarray(ts_logvar_chunk.tolist()).mean(axis=0),
                'mu':np.asarray(ts_mu_chunk.tolist()).mean(axis=0)}

    def training_epoch_end(self, outputs):
        print("Saving MCE before KLD is applied...")

        avg_logvar = np.array([x['logvar'] for x in outputs]).mean(axis=0)
        avg_var = np.array([x['var'] for x in outputs]).mean(axis=0)
        avg_mu = np.array([x['mu'] for x in outputs]).mean(axis=0)


        self.np_logvar_train = np.append(self.np_logvar_train, [avg_logvar],axis=0)
        self.np_var_train = np.append(self.np_var_train, [avg_var],axis=0)
        self.np_mu_train = np.append(self.np_mu_train, [avg_mu],axis=0)


        if(self.current_epoch == self.sigmoid_annealing_threshold ):
            utils.save_dict_as_json(self.mce_batch_train, 'mce_results_wo_kld.json', self.experiment_path_train)
        return {}

    def test_step(self, batch, batch_idx):
        print('test step')

        batch_mce =0
        test_loss = 0

        # self.eval()
        ts_batch_user_features = batch.view(-1, self.input_dimension)
        if (self.mixup):
            ts_batch_user_features, y_a, y_b, lam = training_utils.mixup_data(ts_batch_user_features, self.test_y_bin,
                                                                    alpha=1.0, use_cuda=False)

        recon_batch, ts_mu_chunk, ts_logvar_chunk, p, q = self.forward(ts_batch_user_features)
        ls_z = self.compute_z(ts_mu_chunk, ts_logvar_chunk)
        # ls_z = self.reparameterize_wo_eps(ts_mu_chunk, ts_logvar_chunk)

        self.ls_predicted_movies.extend((-recon_batch).argsort()[:,0].tolist())

        self.np_z_test = np.append(self.np_z_test, np.asarray(ls_z), axis=0) #TODO get rid of np_z_chunk and use np.asarray(mu_chunk)

        self.np_mu_test = np.append(self.np_mu_test, np.asarray(ts_mu_chunk), axis =0)
        self.np_logvar_test = np.append(self.np_logvar_test, np.asarray(ts_logvar_chunk), axis =0)
        # self.np_z = np.vstack((self.np_z, np_z_chunk))

        batch_rmse_w_zeros, batch_mse_w_zeros, batch_rmse, batch_mse = self.calculate_batch_metrics(recon_batch=recon_batch, ts_batch_user_features =ts_batch_user_features)
        batch_mse, kld = self.loss_function(recon_batch,
                                            ts_batch_user_features, #ts_batch_user_features,
                                            ts_mu_chunk,
                                            ts_logvar_chunk,
                                            self.beta,
                                            self.unique_movies,
                                            p,
                                            q,
                                            new_kld_function=False,
                                            train=False)
        #normalizing reconstruction loss
        batch_mse = batch_mse/len(ts_batch_user_features)
        batch_loss = batch_mse + kld

        mce_minibatch = metric_utils.mce_batch(self, ts_batch_user_features, self.dct_index2itemId, k=1)
        self.mce_batch_test = self.average_mce_batch(self.mce_batch_test, mce_minibatch)


        loss = batch_loss.item() / len(ts_batch_user_features)

        self.ls_kld_test.append(self.KLD.tolist())


        tensorboard_logs = {'KLD-Test': kld,
                            'MSE-test': batch_mse}

        return {'test_loss': loss,
                'rmse': batch_rmse,
                'mse': batch_mse,
                'rmse_w_zeros': batch_rmse_w_zeros,
                'mse_w_zeros': batch_mse_w_zeros,
                'log':tensorboard_logs,
                'var': np.exp(np.asarray(ts_logvar_chunk.tolist())).mean(axis=0),
                'mean': np.asarray(ts_mu_chunk.tolist()).mean(axis=0),
                'KLD-Test': kld,
                  'MSE-Test': batch_mse
                }

        # test_loss /= len(test_loader.dataset)
        # print('====> Test set loss: {:.4f}'.format(test_loss))

    def combine_movies_with_z_values(self):
        df_syn_movies = pd.read_csv('../data/generated/syn.csv')
        df_ordered_movies = pd.DataFrame(columns=df_syn_movies.columns)

        for index, id in enumerate(self.ls_predicted_movies):
            df_ordered_movies = df_ordered_movies.append(df_syn_movies.loc[df_syn_movies['id'] == id])
        self.df_movies_z_combined = pd.concat(
            [pd.DataFrame(data=self.np_z_test).reset_index(), df_ordered_movies.reset_index()], axis=1)

    def test_epoch_end(self, outputs):
        # avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.combine_movies_with_z_values()

        avg_loss = np.array([x['test_loss'] for x in outputs]).mean()
        mse_test = np.array([x['MSE-Test'] for x in outputs])
        kld_test =np.array([x['KLD-Test'] for x in outputs])
        avg_var = np.array([x['var'] for x in outputs]).mean(axis=0)
        avg_mean = np.array([x['mean'] for x in outputs]).mean(axis=0)
        self.np_var_test = np.append(self.np_var_test, [avg_var], axis=0)

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

    def new_kld_func(self, p, q):
        log_qz = q.log_prob(self.z)
        log_pz = p.log_prob(self.z) #Normalverteilung

        kl = log_qz - log_pz

        return kl

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, beta, unique_movies, p, q, new_kld_function=False, train=True):
        """ELBO assuming entries of x are binary variables, with closed form KLD."""
        try:
            MSE = F.binary_cross_entropy(recon_x, x, reduction='sum')# MSE is bad for this
            # MSE = F.mse_loss(x, recon_x, reduction='sum')# MSE is bad for this

            if(np.isnan(np.sum(MSE.detach().numpy()))):
                print('MSE tensor is nan')
        except RuntimeError as e:
            print('RuntimeError in loss function', e)
        if(new_kld_function):
            # 1)
            kl = self.new_kld_func(p,q)
            self.kld_matrix_train = np.append(self.kld_matrix_train, np.asarray(kl.tolist()), axis=0)
            kld_mean = kl.mean(dim=0).mean()
        else:
            # see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 --> https://arxiv.org/abs/1312.6114
            #2)
            kld_latent_factors = torch.exp(logvar) + mu ** 2 - 1. - logvar
            kld_mean = -0.5 * torch.mean(torch.sum(-kld_latent_factors, dim=1)) #0: sum over latent factors (columns), 1: sum over sample (rows)

            if(train):
                self.kld_matrix_train = np.append(self.kld_matrix_train, np.asarray(kld_latent_factors.tolist()), axis=0)
            else:
                self.kld_matrix_test = np.append(self.kld_matrix_test, np.asarray(kld_latent_factors.tolist()), axis=0)

        print('KLD true: ', kld_mean)
        print('BCE: ', MSE)
        if(self.training):
            # kld_weight = self.sigmoid_annealing(beta,self.current_epoch)
            kld_weight = beta

        else:
            kld_weight = beta
        # print('kld weight: {}'.format(kld_weight))
        self.KLD = kld_mean * kld_weight

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

            rmse, mse = metric_utils.calculate_metrics(np_y, np_yhat)
            batch_mse += mse
            batch_rmse += rmse

            if(len(np_yhat_wo_zeros)>0):
                rmse_wo_zeros, mse_wo_zeros = metric_utils.calculate_metrics(np_y_wo_zeros, np_yhat_wo_zeros)
                batch_rmse_wo_zeros += rmse_wo_zeros
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
        self.np_logvar_train = dct_attributes['np_logvar_train']
        self.np_var_train = dct_attributes['np_var_train']
        self.np_mu_train = dct_attributes['np_mu_train']
        # self.kld_matrix_train = dct_attributes['kld_matrix_train']
        if(self.np_synthetic_data is not None):
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
        if(self.np_synthetic_data is None):
            dct_attributes = {'np_z_train': self.np_z_train,
                              'kld_matrix_train': self.kld_matrix_train,
                              'np_logvar_train': self.np_logvar_train,
                              'np_var_train': self.np_var_train,
                              'np_mu_train': self.np_mu_train,
                              'ls_kld': self.ls_kld,
                              'z_max_train': self.z_max_train,
                              'z_min_train': self.z_min_train,
                              'z_mean_train': self.z_mean_train}
        else:
            dct_attributes = {'np_z_train':self.np_z_train,
                              'kld_matrix_train': self.kld_matrix_train,
                              'np_logvar_train': self.np_logvar_train,
                              'np_var_train': self.np_var_train,
                              'np_mu_train': self.np_mu_train,
                              'train_y': self.train_y,
                              'test_y':self.test_y,
                              'ls_kld':self.ls_kld,
                              'z_max_train': self.z_max_train,
                              'z_min_train': self.z_min_train,
                              'z_mean_train': self.z_mean_train}
        with open(path, 'wb') as handle:
            pickle.dump(dct_attributes, handle)
        print('Attributes saved')

def generate_distribution_df():
    dct_attribute_distribution = utils.compute_relative_frequency(
        pd.read_csv('../data/generated/syn.csv'))
    utils.save_dict_as_json(dct_attribute_distribution, 'syn_attribute_distribution.json')


#TODO If MC than change title of all plots from MCE to MC

if __name__ == '__main__':
    # Architecture Parameters
    torch.manual_seed(100)
    args = training_utils.create_training_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #  use gpu if available
    settings.init()
    #------------------------

    # General Parameters
    train = False
    mixup = False
    is_hessian_penalty_activated = False
    base_path = 'results/models/vae/'
    used_data = 'vae'
    synthetic_data = True
    full_test_routine = False
    # ------------------------

    #MovieLens Data Parameters
    small_movie_dataset=False
    # ------------------------

    # Synthetic Data Parameters
    expanded_user_item = False
    ls_normalvariate = [False]
    ls_continous = [True]
    noise = False
    no_generative_factors = 3
    # ------------------------

    ls_epochs = [11]
    ls_latent_factors = [5]
    # beta_normalized = 10 / (20 * no_generative_factors)
    ls_betas = [] #disentangle_factors

    for epoch in ls_epochs:
        for normalvariate in ls_normalvariate:
            for continous_data in ls_continous:
                 for lf in ls_latent_factors:
                    if(len(ls_betas)==0):
                        if(expanded_user_item):
                            beta_normalized = lf/(800)
                        else:
                            if(synthetic_data):
                                beta_normalized = lf / (20 * no_generative_factors) #lf/input_size, e.g. 2/10000 = 0.0002
                            else:
                                beta_normalized = lf / 14000#(20 * no_generative_factors) #lf/input_size, e.g. 2/10000 = 0.0002
                        ls_betas.append(beta_normalized)
                    for beta in ls_betas:
                        train_tag = "train"
                        if(not train):
                            train_tag = "test"

                        print("Processing model with: {} epochs, {} latent factors, {} beta".format(epoch, lf, beta))
                        # exp_name = "{}_beta_{}_epochs_{}_lf_synt_{}_normal_{}_continous_{}_hessian_{}_noise_{}".format(beta, epoch, lf, synthetic_data, normalvariate, continous_data, is_hessian_penalty_activated, noise)
                        exp_name = "{}_beta_{}_epochs_{}_lf_synt_{}_normal_{}_continous_{}_hessian_{}".format(beta, epoch, lf,
                                                                                                              synthetic_data,
                                                                                                              normalvariate, continous_data,
                                                                                                              is_hessian_penalty_activated)
                        wandb_name = exp_name + "_" + train_tag
                        model_name = exp_name + ".ckpt"
                        attribute_name = exp_name + "_attributes.pickle"
                        model_path = base_path + model_name
                        attribute_path = base_path + attribute_name

                        experiment_path = utils.create_experiment_directory()

                        model_params = training_utils.create_model_params(experiment_path, epoch, lf,
                                                                          beta, int(epoch / 100),
                                                                          expanded_user_item, mixup,
                                                                          no_generative_factors, epoch,
                                                                          is_hessian_penalty_activated, used_data,
                                                                          small_movie_dataset)

                        args.max_epochs = epoch

                        wandb_logger = WandbLogger(project='recommender-xai', tags=['vae', train_tag], name=wandb_name)
                        trainer = pl.Trainer.from_argparse_args(args,
                                                                # limit_test_batches=0.1,
                                                                # precision =16,
                                                                logger=wandb_logger, #False
                                                                gradient_clip_val=0.5,
                                                                # accumulate_grad_batches=0,
                                                                gpus=0,
                                                                weights_summary='full',
                                                                checkpoint_callback = False,
                                                                callbacks = [ProgressBar(), EarlyStopping(monitor='train_loss')]
                        )

                        if(train):
                            print('<---------------------------------- VAE Training ---------------------------------->')
                            print("Running with the following configuration: \n{}".format(args))
                            if (synthetic_data):
                                model_params['synthetic_data'], model_params['syn_y'] = data_utils.create_synthetic_data(no_generative_factors,
                                                                                                                         experiment_path,
                                                                                                                         expanded_user_item,
                                                                                                                         continous_data,
                                                                                                                         normalvariate,
                                                                                                                         noise)
                                generate_distribution_df()

                            model = VAE(model_params)
                            wandb_logger.watch(model, log='gradients', log_freq=100)

                            # plot_utils.print_nn_summary(model, size =model.np_synthetic_data.shape[1])

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
                        test_model = VAE.load_from_checkpoint(model_path)#, load_saved_attributes=True, saved_attributes_path='attributes.pickle'
                        # test_model.test_size = model_params['test_size']
                        test_model.load_attributes_and_files(attribute_path)
                        test_model.experiment_path_test = experiment_path

                        # print("show np_z_train mean:{}, min:{}, max:{}".format(z_mean_train, z_min_train, z_max_train ))
                        print('------ Start Test ------')
                        start = time.time()
                        dct_param ={'epochs':epoch, 'lf':lf,'beta':beta, 'normal':normalvariate,
                                    'continous':continous_data, 'hessian':is_hessian_penalty_activated, 'noise':noise}
                        # plot_utils.plot_samples(test_model, experiment_path, dct_param)

                        if(test_model.no_latent_factors<11):
                            latent_space_utils.traverse(test_model, experiment_path, dct_param)
                            if(test_model.np_synthetic_data is not None):
                                metric_utils.create_multi_mce(test_model, dct_param)

                        trainer.test(test_model) #The test loop will not be used until you cadct_paramll.
                        print('Test time in seconds: {}'.format(time.time() - start))
                        # print('% altering has provided information gain:{}'.format( int(settings.ig_m_hat_cnt)/(int(settings.ig_m_cnt)+int(settings.ig_m_hat_cnt) )))
                        # print(results)

                        if(synthetic_data):
                            disentangle_utils.run_disentanglement_eval(test_model, experiment_path, dct_param)

                        plot_utils.plot_results(test_model,
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

                        print('Test done')
    exit()



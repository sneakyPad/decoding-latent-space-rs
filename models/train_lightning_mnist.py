# pip install pytorch-lightning
# pip install neptune-client
#%%
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
from disentvaeutils.datasets import get_dataloaders, get_img_size, DATASETS
from utils import disentangle_utils, training_utils, plot_utils, data_utils, utils, metric_utils, settings,io
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

from torchvision import datasets, transforms

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
    def __init__(self, conf:dict, *args, **kwargs):
        super().__init__()

        # self.kwargs = kwargs
        self.save_hyperparameters(conf)
        self.ls_predicted_movies = []
        self.is_hessian_penalty_activated = self.hparams["is_hessian_penalty_activated"]
        self.used_data = self.hparams["used_data"]
        self.expanded_user_item = self.hparams["expanded_user_item"]
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
        self.unique_movies =0
        self.np_user_item = None
        self.small_dataset = self.hparams["small_dataset"]
        self.simplified_rating = self.hparams["simplified_rating"]
        self.max_epochs = self.hparams["max_epochs"]
        self.dct_index2itemId = None
        self.test_y_bin = None
        self.df_movies_z_combined =None

        # if(self.np_synthetic_data is None):
        #     self.load_dataset() #additionaly assigns self.unique_movies and self.np_user_item
        #     self.df_movies = pd.read_csv('../data/generated/df_movies_cleaned3.csv')
        #     self.dct_attribute_distribution = utils.load_json_as_dict(
        #         'attribute_distribution.json')  # load relative frequency distributioon from dictionary (pickle it)
        #
        # else:
        #     self.train_dataset, self.test_dataset = train_test_split(self.np_synthetic_data, test_size=self.test_size, random_state=42)
        #     self.train_y, self.test_y = train_test_split(self.ls_syn_y, test_size=self.test_size, random_state=42)
        #     self.test_y_bin = np.asarray(pd.get_dummies(pd.DataFrame(data=self.test_y)))
        #     self.unique_movies = self.np_synthetic_data.shape[1]
        #     self.df_movies = pd.read_csv('../data/generated/syn.csv')
        #     self.dct_attribute_distribution = utils.load_json_as_dict(
        #         'syn_attribute_distribution.json')  # load relative frequency distributioon from dictionary (pickle it)

        self.bs = 100

        #nn.Linear layer creates a linear function (θx + b), with its parameters initialized
        self.input_dimension = int(self.unique_movies *math.pow(4, self.generative_factors)) if self.expanded_user_item == True else self.unique_movies

        if(self.used_data=='morpho'):
            self.input_dimension = 28*28
        elif(self.used_data=='dsprites'):
            self.input_dimension = 64 * 64

        self.fc11 = nn.Linear(in_features=self.input_dimension, out_features=1200) #input
        self.fc12 = nn.Linear(in_features=1200, out_features=1200) #input
        self.fc13 = nn.Linear(in_features=1200, out_features=1200) #input
        self.encoder = nn.Sequential(self.fc11,
                                     self.fc12,
                                     self.fc13
                                     )

        self.fc21 = nn.Linear(in_features=1200, out_features=self.no_latent_factors) #encoder mean
        self.fc22 = nn.Linear(in_features=1200, out_features=self.no_latent_factors) #encoder variance

        self.fc31 = nn.Linear(in_features=self.no_latent_factors, out_features=1200)
        self.fc32 = nn.Linear(in_features=1200, out_features=1200)
        self.fc33 = nn.Linear(in_features=1200, out_features=1200)
        self.fc34 = nn.Linear(in_features=1200, out_features=self.input_dimension)
        self.decoder = nn.Sequential(self.fc31, self.fc32, self.fc33, self.fc34)

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

        self.sigmoid_annealing_threshold = self.hparams['sigmoid_annealing_threshold']
        self.mce_batch_train = None
        self.mce_batch_test = None

        self.z_mean_train = []
        self.z_min_train = []
        self.z_max_train = []

        # Initialize weights
        self.encoder.apply(self.weight_init)
        self.decoder.apply(self.weight_init)

        self.batch_size =512
        import sklearn
        # np_user_item, ls_y = sklearn.utils.shuffle(np_user_item, ls_y)
        if (self.used_data == 'dsprites'):
            self.batch_size =2048
            loader = get_dataloaders('dsprites', batch_size=512, shuffle=False)
            dsprites_data = loader.dataset.imgs#[:5000]
            self.dsprites_lat_names = loader.dataset.lat_names
            dsprites_gen_fac_values = loader.dataset.lat_values#[:5000]


            # self.train_dataset, self.test_dataset = train_test_split(dsprites_data, test_size=self.test_size, shuffle=False,random_state=42)
            #dsprites_data[:int(dsprites_data.shape[0]*0.1)]
            self.train_dataset, self.test_dataset = train_test_split(dsprites_data,
                                                                     test_size=self.test_size, shuffle=True,random_state=42)
            self.train_y, self.test_y = train_test_split(dsprites_gen_fac_values,
                                                         test_size=self.test_size, shuffle=True,random_state=42)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # nn.init.orthogonal_(m.weight)
            m.bias.data.zero_()

    def encode(self, x):
        h1 = F.relu(self.encoder(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

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

    def train_dataloader(self):

        # MNIST Dataset


        if(self.used_data == 'morpho'):
            MORPHO_MNIST_FILE_TRAIN_Y = "/Users/d069735/workspace/Study/decoding-latent-space-rs/data/morpho-mnist/global/train-pert-idx1-ubyte.gz"
            MORPHO_MNIST_FILE_TRAIN_X = "/Users/d069735/workspace/Study/decoding-latent-space-rs/data/morpho-mnist/global/train-images-idx3-ubyte.gz"
            self.train_dataset = io.load_idx(MORPHO_MNIST_FILE_TRAIN_X)
            self.train_y = io.load_idx(MORPHO_MNIST_FILE_TRAIN_Y)
        elif(self.used_data == 'dsprites'):
            # train_loader = get_dataloaders('dsprites', batch_size=512, shuffle=False)
            train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, num_workers=8,batch_size=self.batch_size, shuffle=True)
            return train_loader
        #regular mnist
        else:
            self.train_dataset = datasets.MNIST(root='../data/mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
            self.train_y = self.train_dataset.targets.tolist()

        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader

    def test_dataloader(self):
        bar = datasets.MNIST(root='../data/mnist_data/', train=False, transform=transforms.ToTensor(),download=False)

        if(self.used_data == 'morpho'):
            MORPHO_MNIST_FILE_TEST_Y = "/Users/d069735/workspace/Study/decoding-latent-space-rs/data/morpho-mnist/global/t10k-pert-idx1-ubyte.gz"
            MORPHO_MNIST_FILE_TEST_X = "/Users/d069735/workspace/Study/decoding-latent-space-rs/data/morpho-mnist/global/t10k-images-idx3-ubyte.gz"
            self.test_dataset = io.load_idx(MORPHO_MNIST_FILE_TEST_X)
            self.test_y = io.load_idx(MORPHO_MNIST_FILE_TEST_Y)
        elif(self.used_data == 'dsprites'):
            # self.test_dataset =
            test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
            return test_loader
        #regular mnist

        else:
            self.test_y = self.test_dataset.targets.tolist()
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.bs, shuffle=False)

        return test_loader

    def configure_optimizers(self):
        optimizer = optim.Adagrad(self.parameters(), lr=1e-2) #Adam
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

        # print('train step')
        # (data, _) = batch
        if(self.used_data=='morpho'):
            ts_batch_user_features = batch.view(-1, 784).float() / 255. #.unsqueeze(1)
        elif(self.used_data=='dsprites'):
            ts_batch_user_features = batch.view(-1, 64*64).float() / 255. #.unsqueeze(1)
        recon_batch, ts_mu_chunk, ts_logvar_chunk, p, q = self.forward(ts_batch_user_features)  # sample data
        if(self.current_epoch == self.max_epochs-1):
            # print("Last round..")
            self.collect_z_values(ts_mu_chunk, ts_logvar_chunk)#, ls_preference

        if (self.current_epoch == self.sigmoid_annealing_threshold ):
            self.collect_z_values(ts_mu_chunk, ts_logvar_chunk)
            # mce_minibatch = mce_batch(self, ts_batch_user_features, self.dct_index2itemId, k=3)
            # self.mce_batch_train = self.average_mce_batch(self.mce_batch_train, mce_minibatch)

        batch_mse, batch_kld = self.loss_function(recon_batch,
                                                  ts_batch_user_features, #ts_batch_user_features,
                                                  ts_mu_chunk,
                                                  ts_logvar_chunk,
                                                  self.beta,
                                                  self.unique_movies,
                                                  p,
                                                  q,
                                                  new_kld_function = True)
        hp_loss =0
        if(self.is_hessian_penalty_activated and self.current_epoch > int(1/3*self.max_epochs-1)):
            print('<---- Applying Hessian Penalty ---->')
            np_z = self.compute_z(ts_mu_chunk, ts_logvar_chunk)
            hp_loss = hessian_penalty(G=self.decode, z=np_z)
            print('Hessian Penalty:{}'.format(hp_loss))
            batch_loss = hp_loss
        else:

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
        # print('test step')

        batch_mce =0
        test_loss = 0

        # self.eval()
        # ts_batch_user_features = batch.view(-1, self.input_dimension)
        # (data, _) = batch
        if (self.used_data == 'morpho'):
            ts_batch_user_features = batch.view(-1, 784).float() / 255.
        elif (self.used_data == 'dsprites'):
            ts_batch_user_features = batch.view(-1, 64*64).float() / 255.

        if (self.mixup):
            ts_batch_user_features, y_a, y_b, lam = self.mixup_data(ts_batch_user_features, self.test_y_bin,
                                                                    alpha=1.0, use_cuda=False)

        recon_batch, ts_mu_chunk, ts_logvar_chunk, p, q = self.forward(ts_batch_user_features)
        ls_z = self.compute_z(ts_mu_chunk, ts_logvar_chunk)

        self.ls_predicted_movies.extend((-recon_batch).argsort()[:,0].tolist())

        self.np_z_test = np.append(self.np_z_test, np.asarray(ls_z), axis=0) #TODO get rid of np_z_chunk and use np.asarray(mu_chunk)

        self.np_mu_test = np.append(self.np_mu_test, np.asarray(ts_mu_chunk), axis =0)
        self.np_logvar_test = np.append(self.np_logvar_test, np.asarray(ts_logvar_chunk), axis =0)
        # self.np_z = np.vstack((self.np_z, np_z_chunk))

        # batch_rmse_w_zeros, batch_mse_w_zeros, batch_rmse, batch_mse = self.calculate_batch_metrics(recon_batch=recon_batch, ts_batch_user_features =ts_batch_user_features)
        batch_mse, kld = self.loss_function(recon_batch,
                                            ts_batch_user_features, #ts_batch_user_features,
                                            ts_mu_chunk,
                                            ts_logvar_chunk,
                                            self.beta,
                                            self.unique_movies,
                                            p,
                                            q,
                                            new_kld_function=True)
        batch_loss = batch_mse + kld

        # mce_minibatch = mce_batch(self, ts_batch_user_features, self.dct_index2itemId, k=3)
        # self.mce_batch_test = self.average_mce_batch(self.mce_batch_test, mce_minibatch)

        #to be rermoved mean_mce = { for single_mce in batch_mce}
        loss = batch_loss.item() / len(ts_batch_user_features)

        # bce = batch_bce/len(ts_batch_user_features)
        tensorboard_logs = {'KLD-Test': kld,
                            'MSE-test': batch_mse}

        return {'test_loss': loss,
                # 'rmse': batch_rmse,
                'mse': batch_mse,
                # 'rmse_w_zeros': batch_rmse_w_zeros,
                # 'mse_w_zeros': batch_mse_w_zeros,
                'log':tensorboard_logs,
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
        # ls_mce = {x['mce'] for x in outputs}
        # utils.save_dict_as_json(self.mce_batch_test, 'mce_results.json', self.experiment_path_test)
        # avg_mce = dict(calculate_mean_of_ls_dict(ls_mce))

        # avg_rmse = np.array([x['rmse'] for x in outputs]).mean()
        # avg_rmse_w_zeros = np.array([x['rmse_w_zeros'] for x in outputs]).mean()
        # avg_mse = np.array([x['mse'] for x in outputs]).mean()
        # avg_mse_w_zeros = np.array([x['mse_w_zeros'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss, 'MSE-Test':mse_test,'KLD-Test': kld_test }
        assert len(mse_test)==len(kld_test)
        for i in range(0, len(mse_test)):
            wandb_logger.log_metrics({'MSE-Test': mse_test[i],'KLD-Test': kld_test[i]} )

        # wandb_logger.log_metrics({'rmse': avg_rmse,
        #                           'rmse_w_zeros':avg_rmse_w_zeros,
        #                           'mse': avg_mse,
        #                           'mse_w_zeros': avg_mse_w_zeros})#, 'kld_matrix':self.kld_matrix

        return {'test_loss': avg_loss,
                'log': tensorboard_logs,
                # 'rmse': avg_rmse,
                'MSE-Test':mse_test,
                'KLD-test': kld_test }#, , 'mce':avg_mce

    def sigmoid_annealing(self, beta, epoch):

        stretch_factor = 0.5
        if(epoch < self.sigmoid_annealing_threshold):
            return 0
        else:
            kld_weight = beta/(1+ math.exp(-epoch * stretch_factor + self.sigmoid_annealing_threshold)) #epoch_threshold moves e function along the x-axis
            return kld_weight

    def kl_divergence(self,p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))


    # def step(self, batch, batch_idx):
    #     x, y = batch
    #     z, x_hat, p, q = self._run_step(x)
    #
    #     recon_loss = F.mse_loss(x_hat, x, reduction='mean')
    #
    #     log_qz = q.log_prob(z)
    #     log_pz = p.log_prob(z)
    #
    #     kl = log_qz - log_pz
    #     kl = kl.mean()

    def new_kld_func(self, p, q):
        log_qz = q.log_prob(self.z)
        log_pz = p.log_prob(self.z)

        kl = log_qz - log_pz

        return kl

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, beta, unique_movies, p, q, new_kld_function=False):


        MSE = F.binary_cross_entropy(recon_x, x,reduction='sum')  # TODO: Is that correct? binary cross entropy - (Encoder)
        # MSE = F.mse_loss(x, recon_x)#/x.shape[1]

        # if(new_kld_function):
        #     kl = self.new_kld_func(p,q)
        #     self.kld_matrix = np.append(self.kld_matrix, np.asarray(kl.tolist()), axis=0)
        #     kld_mean = kl.mean()
        # else:
            # see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114

            # p = norm.pdf(recon_x[:,0].tolist(), mu[:,0].tolist(), logvar[:,0].tolist())
            # q = norm.pdf(x[:,0].tolist(), 0, 1)
            # kl_medium = self.kl_divergence(p,q)

        #     kld_latent_factors = torch.exp(logvar) + mu ** 2 - 1. - logvar
        #     kld_mean = -0.5 * torch.mean(torch.sum(-kld_latent_factors, dim=1)) #0: sum over latent factors (columns), 1: sum over sample (rows)
        #     self.kld_matrix = np.append(self.kld_matrix, np.asarray(kld_latent_factors.tolist()), axis=0)
        #
        # if(self.training):
        #     kld_weight = self.sigmoid_annealing(beta,self.current_epoch)
        # else:
        #     kld_weight = beta
        #
        # self.KLD = kld_mean * kld_weight

        kl= -(1 + logvar - mu.pow(2) - logvar.exp())
        self.kld_matrix = np.append(self.kld_matrix, np.asarray(kl.tolist()), axis=0)
        self.KLD = 0.5 * torch.sum(kl)
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
        self.train_y = dct_attributes['train_y']
        self.ls_kld = dct_attributes['ls_kld']

        # self.dct_attribute_distribution = utils.load_json_as_dict(
        #     'attribute_distribution.json')  # load relative frequency distributioon from dictionary (pickle it)

        self.z_max_train = dct_attributes['z_max_train']
        self.z_min_train = dct_attributes['z_min_train']
        self.z_mean_train = dct_attributes['z_mean_train']
        print('Attributes loaded')

    def save_attributes(self, path):
        dct_attributes = {'np_z_train':self.np_z_train,
                          'train_y': self.train_y,
                          'ls_kld':self.ls_kld,
                          'z_max_train': self.z_max_train,
                          'z_min_train': self.z_min_train,
                          'z_mean_train': self.z_mean_train}
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
    #TODO Source read_csv out
    # df_links = pd.read_csv('../data/movielens/small/links.csv')
    # df_movies = pd.read_csv('../data/generated/df_movies_cleaned3.csv')


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


def alter_z(ts_z, latent_factor_position, model, strategy):
    if(strategy == 'max'):
        ts_z[:, latent_factor_position] = model.z_max_train[latent_factor_position] #32x no_latent_factors, so by accesing [:,pos] I get the latent factor for the batch of 32 users
    elif(strategy == 'min'):
        raise NotImplementedError("Min Strategy needs to be implemented")
    elif(strategy == 'min_max'):
        try:
            z_max_range = model.z_max_train[latent_factor_position] #TODO Evtl. take z_max_train here
            z_min_range = model.z_min_train[latent_factor_position]

            if(np.abs(z_max_range) > np.abs(z_min_range)):
                ts_z[:, latent_factor_position] = model.z_max_train[latent_factor_position]
            else:
                ts_z[:, latent_factor_position] = model.z_min_train[latent_factor_position]
        except IndexError:
            print('stop')
    return ts_z

#MCE is calculated for each category
def mce_batch(model, ts_batch_features, dct_index2itemId, k=0):
    dct_mce_mean = defaultdict()
    # hold n neurons of hidden layer
    # change 1 neuron
    ls_y_hat, mu, logvar, p, q = model(ts_batch_features)
    z = model.z
    # dct_attribute_distribution = utils.load_json_as_dict('attribute_distribution.json') #    load relative frequency distributioon from dictionary (pickle it)

    for latent_factor_position in range(model.no_latent_factors):

        # print("Calculate MCEs for latent factor: {}".format(latent_factor_position))
        ts_altered_z = alter_z(z, latent_factor_position, model, strategy='min_max')#'max'

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
        #Iterate through batches of recommended items by user vector
        for user_idx in range(len(ls_idx_yhat)): #tqdm(range(len(ls_idx_yhat)), total = len(ls_idx_yhat)):
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
            single_mce = metric_utils.mce_information_gain(y_hat_w_metadata, y_hat_latent_w_metadata, model.dct_attribute_distribution) #mce for n columns
            ls_dct_mce.append(single_mce)

            # print(single_mce)
        dct_mce_mean[latent_factor_position] = dict(calculate_mean_of_ls_dict(ls_dct_mce))
    return dct_mce_mean


def generate_distribution_df():
    dct_attribute_distribution = utils.compute_relative_frequency(
        pd.read_csv('../data/generated/syn.csv'))
    utils.save_dict_as_json(dct_attribute_distribution, 'syn_attribute_distribution.json')

if __name__ == '__main__':
    torch.manual_seed(100)
    args = training_utils.create_training_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #  use gpu if available
    settings.init()


    # print(data)
    # io.save_idx(data, TEST_FILE)
    # data_ = io.load_idx(TEST_FILE)
    #
    # import os
    #
    # os.remove(TEST_FILE)
    #
    # assert np.alltrue(data == data_)



    #%%
    train = True
    synthetic_data = True
    expanded_user_item = False
    mixup = False
    is_hessian_penalty_activated = False
    continous_data=False
    normalvariate = False
    # morpho = True
    # used_data = "morpho"
    used_data = "dsprites"
    base_path = 'results/models/vae/'

    ls_epochs = [5]#5 with new data, 70 was trained w/ old mnist
    #Note: Mit steigender Epoche wird das disentanglement verstärkt
    #
    ls_latent_factors = [10]
    ls_betas = [4] #disentangle_factors .0003
    no_generative_factors = 3

    for epoch in ls_epochs:
        for lf in ls_latent_factors:
            if(len(ls_betas)==0):
                if(expanded_user_item):
                    beta_normalized = lf/(800)
                else:
                    beta_normalized = lf / (20 * no_generative_factors) #lf/input_size, e.g. 2/10000 = 0.0002
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

                model_params = training_utils.create_model_params(experiment_path, epoch, lf, beta, int(epoch / 2), expanded_user_item, mixup,
                        no_generative_factors, epoch, is_hessian_penalty_activated, used_data)

                args.max_epochs = epoch

                wandb_logger = WandbLogger(project='recommender-xai', tags=['vae', train_tag], name=wandb_name)
                trainer = pl.Trainer.from_argparse_args(args,
                                                        logger=wandb_logger, #False
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
                                                                                                                 normalvariate)
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
                #Sample
                # z = torch.randn(64, lf)
                # sample = model.decode(z)
                # save_image(sample.view(64, 1, 28, 28), './results/mnist_imgs/sample_morpho_20' + '.png')


                test_model = VAE.load_from_checkpoint(model_path)#, load_saved_attributes=True, saved_attributes_path='attributes.pickle'
                # test_model.test_size = model_params['test_size']
                test_model.load_attributes_and_files(attribute_path)
                test_model.experiment_path_test = experiment_path

                # print("show np_z_train mean:{}, min:{}, max:{}".format(z_mean_train, z_min_train, z_max_train ))
                print('------ Start Test ------')
                start = time.time()
                trainer.test(test_model) #The test loop will not be used until you call.
                # print('Test time in seconds: {}'.format(time.time() - start))
                # print('% altering has provided information gain:{}'.format( int(settings.ig_m_hat_cnt)/(int(settings.ig_m_cnt)+int(settings.ig_m_hat_cnt) )))
                # print(results)

                dct_param ={'epochs':epoch, 'lf':lf,'beta':beta}

                plot_utils.plot_results(test_model,
                                   test_model.experiment_path_test,
                                   test_model.experiment_path_train,
                                   dct_param )

                disentangle_utils.run_disentanglement_eval(test_model, experiment_path, dct_param)

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


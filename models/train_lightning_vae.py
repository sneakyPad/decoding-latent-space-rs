from __future__ import print_function
import torch, torch.nn as nn, torchvision, torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
import recmetrics
# import matplotlib.pyplot as plt
from surprise import Reader, SVD, Dataset
# from surprise.model_selection import train_test_split

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torchvision.utils import save_image

import pytorch_lightning as pl
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


#%%
# pip install pytorch-lightning
# pip install neptune-client
#%%

##This method creates a user-item matrix by transforming the seen items to 1 and adding unseen items as 0 if simplified_rating is set to True
##If set to False, the actual rating is taken
##Shape: (n_user, n_items)
unique_movies = 0
def create_user_item_matrix(df, simplified_rating: bool):
    # unique_movies = len(df["movieId"].unique())
    global unique_movies
    unique_movies = df["movieId"].unique().max() + 1
    ls_users = df["userId"].unique()
    unique_users = len(df["userId"].unique()) +1
    np_user_item_mx = np.zeros((unique_users,unique_movies),dtype=np.float_)

    for user_id in ls_users:
        if(user_id%100 == 0):
            print("User-Id: {} ".format(user_id))
        ls_seen_items = df.loc[df["userId"] == user_id]["movieId"].values
        if(simplified_rating):
            np_user_item_mx[user_id][ls_seen_items] = 1
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
    return np_user_item_mx.astype(np.float32), unique_movies


class VAE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.test_size = kwargs["test_size"]
        self.unique_movies =  0
        self.np_user_item = None
        self.small_dataset = kwargs["small_dataset"]
        self.simplified_rating = kwargs["simplified_rating"]


        self.load_dataset() #additionaly assigns self.unique_movies and self.np_user_item


        #nn.Linear layer creates a linear function (θx + b), with its parameters initialized
        self.fc1 = nn.Linear(in_features=self.unique_movies, out_features=400) #input
        self.fc21 = nn.Linear(in_features=400, out_features=20) #encoder mean
        self.fc22 = nn.Linear(in_features=400, out_features=20) #encoder variance
        self.fc3 = nn.Linear(in_features=20, out_features=400)
        self.fc4 = nn.Linear(in_features=400, out_features=self.unique_movies)

        # self.save_hyperparameters()

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

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.unique_movies))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def load_dataset(self):
        if (self.small_dataset):
            print("Load small dataset")
            df_ratings = pd.read_csv("../data/openlens/small/ratings.csv")
        else:
            print("Load large dataset")
            df_ratings = pd.read_csv("../data/openlens/large/ratings.csv")

        self.np_user_item, self.unique_movies = create_user_item_matrix(df_ratings, simplified_rating=self.simplified_rating)
        self.train_dataset, self.test_dataset = train_test_split(self.np_user_item, test_size=self.test_size, random_state=42)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=128, shuffle=True, num_workers=1, pin_memory=True
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

    # def step(self, ts_batch_user_features):


    def training_step(self, batch, batch_idx):
        ts_batch_user_features = batch
        recon_batch, mu, logvar = self.forward(ts_batch_user_features)  # sample data
        batch_loss = loss_function(recon_batch, ts_batch_user_features, mu, logvar, unique_movies)
        loss = batch_loss/len(ts_batch_user_features)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}


    # def validation_step(self, batch, batch_idx):
    #     return 0
    #
    def test_step(self, batch, batch_idx):
        model.eval()
        ts_batch_user_features = batch

        test_loss = 0
        # with torch.no_grad():
        #     for i, data in enumerate(test_loader):

        recon_batch, mu, logvar = model(ts_batch_user_features)
        batch_loss = loss_function(recon_batch, ts_batch_user_features, mu, logvar, unique_movies).item()
        loss = batch_loss / len(ts_batch_user_features)

        return {'test_loss': batch_loss}

        # test_loss /= len(test_loader.dataset)
        # print('====> Test set loss: {:.4f}'.format(test_loss))

    def test_epoch_end(self, outputs):
        # avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_loss = np.array([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'test_log': tensorboard_logs}


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, unique_movies):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, unique_movies), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


    #Input: All seen items of a user as vector
    # def prediction_user(self, user_feature):
    #     user_preds = self.forward(user_feature)
    #
    #     return user_preds
    #
    # #Input: One Hot Encoding
    # def prediction_single(self, one_hot_movie_id):
    #     pred = self.forward(one_hot_movie_id)
    #     return pred

train_dataset = None
test_dataset = None
# def train(dct_hyperparam: dict, simplified_rating: bool, small_dataset: bool):
#
#     model = VAE(input_shape=unique_movies).to(device)  # load it to the specified device, either gpu or cpu
#     # optimizer = optim.Adam(model.parameters(), lr= dct_hyperparam['learning_rate']) # create an optimizer object -Adam optimizer with learning rate 1e-3
#
#
#
#     ## VAE IMPORTED ---
#     model.train()
#     for epoch in range(dct_hyperparam["epochs"]):
#         loss = 0
#         y_hat = []
#         batch_idx = 0
#         train_loss = 0
#
#         for batch_idx, ts_batch_user_features in enumerate(train_loader):
#             ts_batch_user_features = ts_batch_user_features.to(device)
#             optimizer.zero_grad()
#
#             recon_batch, mu, logvar = model(ts_batch_user_features) #sample data
#             loss = loss_function(recon_batch, ts_batch_user_features, mu, logvar, unique_movies)
#
#
#             loss.backward()
#             batch_loss = loss.item()
#             loss = batch_loss / len(ts_batch_user_features)
#             train_loss += loss
#             optimizer.step()
#             if batch_idx % args.log_interval == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(ts_batch_user_features), len(train_loader.dataset),
#                            100. * batch_idx / len(train_loader),
#                     loss))
#
#             # avg_epoch_loss = train_loss / len(train_loader) Smooth, but weird numbers
#         avg_epoch_loss = train_loss / batch_idx
#
#         writer.add_scalar('training loss', avg_epoch_loss, epoch)
#         print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_epoch_loss))

    ## VAE END IMPORTED ---


# def test():
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, data in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar, unique_movies).item()
#
#
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

#%%
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--max_epochs', type=int, default=2, metavar='N',
                    help='number of max epochs to train (default: 15)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=1, metavar='N',
#                     help='how many batches to wait before logging training status')
args = parser.parse_args()


torch.manual_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #  use gpu if available


#%%

from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning.logging.neptune import NeptuneLogger

# default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('runs/vae_1')
# dct_hyperparam = {
# "batch_size" : 512,
# "epochs" : 20,
# "learning_rate" : 1e-3,
# "k": 500
# }
import recmetrics

#%%
model_params = {"simplified_rating": True,
                "small_dataset": True,
                "test_size": 0.33}
# model_params.update(args.__dict__)
# print(**model_params)

merged_params = (lambda first_dict, second_dict: {**first_dict, **second_dict})(args.__dict__, model_params)
print(merged_params)


#%%


neptune_logger = NeptuneLogger(
    #api_key="key",
    project_name="paer/recommender-xai",
    experiment_name="default_2",  # Optional,
    params = merged_params,
    # params={"max_epochs": 1,
    #         "batch_size": 32},  # Optional,
    close_after_fit=True, #must be False if test method should be logged
    tags=["pytorch-lightning", "vae"]  # Optional,

)

trainer = pl.Trainer.from_argparse_args(args,
                                        #max_epochs=1, automatically processed by Pytorch Light
                                        logger=neptune_logger,
                                        gpus=0,
                                        #num_nodes=4
)




#%%

print("Running with the following configuration: \n{}".format(args))
model = VAE(**model_params)
trainer.fit(model)
trainer.test(model) #The test loop will not be used until you call.

#%%
# plot_ae_img(batch_features,test_loader)
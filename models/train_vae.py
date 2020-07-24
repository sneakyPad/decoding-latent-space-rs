from __future__ import print_function
import torch, torch.nn as nn, torchvision, torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from torchvision import datasets, transforms
from torchvision.utils import save_image


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

##This method creates a user-item matrix by transforming the seen items to 1 and adding unseen items as 0 if simplified_rating is set to True
##If set to False, the actual rating is taken
##Shape: (n_user, n_items)
unique_items = 0
def create_user_item_matrix(df, simplified_rating: bool):
    # unique_movies = len(df["movieId"].unique())
    global unique_items
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
    return np_user_item_mx.astype(np.float32)


#%%
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


torch.manual_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #  use gpu if available

class VAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_shape = kwargs["input_shape"]
        #nn.Linear layer creates a linear function (θx + b), with its parameters initialized
        self.fc1 = nn.Linear(in_features= self.input_shape, out_features=400) #input
        self.fc21 = nn.Linear(in_features=400, out_features=20) #encoder mean
        self.fc22 = nn.Linear(in_features=400, out_features=20) #encoder variance
        self.fc3 = nn.Linear(in_features=20, out_features=400)
        self.fc4 = nn.Linear(in_features=400, out_features= self.input_shape)

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
        mu, logvar = self.encode(x.view(-1, self.input_shape))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



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




# model = VAE().to(device)

train_dataset = None
test_dataset = None
def train(dct_hyperparam: dict, simplified_rating: bool, small_dataset: bool):
    df_ratings = None
    global model
    global train_loader
    # global optimizer
    epoch = dct_hyperparam["epochs"]
    if (small_dataset):
        print("Load small dataset")
        df_ratings = pd.read_csv("../data/openlens/small/ratings.csv")
    else:
        print("Load large dataset")
        df_ratings = pd.read_csv("../data/openlens/large/ratings.csv")

    print("Training started...")
    np_user_item = create_user_item_matrix(df_ratings, simplified_rating=simplified_rating)
    global train_dataset
    global test_dataset
    global test_loader
    train_dataset, test_dataset = train_test_split(np_user_item, test_size=0.33, random_state=42)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=1, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=1
    )

    model = VAE(input_shape=unique_items).to(device)  # load it to the specified device, either gpu or cpu
    # optimizer = optim.Adam(model.parameters(), lr= dct_hyperparam['learning_rate']) # create an optimizer object -Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  # mean-squared error loss


    ## VAE IMPORTED ---
    model.train()
    for epoch in range(dct_hyperparam["epochs"]):
        loss = 0
        y_hat = []
        batch_idx = 0
        train_loss = 0

        for batch_idx, ts_batch_user_features in enumerate(train_loader):
            ts_batch_user_features = ts_batch_user_features.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(ts_batch_user_features) #sample data
            loss = loss_function(recon_batch, ts_batch_user_features, mu, logvar, unique_items)
            loss.backward()
            batch_loss = loss.item()
            loss = batch_loss / len(ts_batch_user_features)
            train_loss += loss
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(ts_batch_user_features), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss))

            # avg_epoch_loss = train_loss / len(train_loader) Smooth, but weird numbers
        avg_epoch_loss = train_loss / batch_idx

        writer.add_scalar('training loss', avg_epoch_loss, epoch)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_epoch_loss))

    ## VAE END IMPORTED ---



    # #TODO fix broken pipe line caused by recmetrics
    #     # display the epoch training loss
    #     print("\nEpoch : {}/{}, Reconstruction-Loss = {:.8f}".format(epoch + 1, dct_hyperparam["epochs"], loss))
    #
    #     mse = recmetrics.mse(train_dataset.tolist(), y_hat)
    #     print("MSE Training Reconstruction: {}".format(mse))
    #
    # #Compute Test Loss
    # mse_user_w_zeros = 0
    # mse_user_wo_zeros = 0
    # mar_k = 0
    # k = dct_hyperparam["k"]
    # with torch.no_grad():
    #     for ts_batch_user_features in test_loader: #testloader holds user-item matrix n,i, e.g. 202,193610 -> 32, 193610
    #         ts_batch_user_features = ts_batch_user_features.to(device)
    #         #Test reconstruction
    #         reconstruction = model(ts_batch_user_features) #compute recommendations
    #         test_loss = criterion(reconstruction, ts_batch_user_features)
    #         loss += test_loss.item()
    #
    #
    #         #Test user prediction
    #         tsls_yhat_user = model.prediction_user(ts_batch_user_features)
    #         #Compute MAR@K
    #         ls_idx_yhat = (-tsls_yhat_user).argsort() #argsort returns indices of the given list in ascending order. For descending we invert the list, so each element is inverted
    #         ls_idx_y = (-ts_batch_user_features).argsort()
    #         mar_k += recmetrics.mark(ls_idx_y.tolist(), ls_idx_yhat.tolist(), k)
    #
    #         #Compute MSE
    #         mask = ts_batch_user_features == tsls_yhat_user #Obtain a mask for filtering out items that haven't been seen nor recommended
    #         tsls_yhat_user = tsls_yhat_user[~mask] #Filter out unseen+unrecommended items
    #         #TODO Mask filters also 1 out, that's bad
    #         ts_user_features_seen = ts_batch_user_features[~mask] #Filter out unseen+unrecommended items
    #         mse_user_w_zeros += recmetrics.mse(ts_user_features_seen.tolist(), tsls_yhat_user) #Compute MSE on items that have been recommended
    #
    #
    #         #Test single prediction
    #
    #         #Snippet is to create a mask for filtering out items that haven't been seen
    #         #not necessary since we can just multiply the output by the input and have a natural filter
    #         # mask_not_null = np.zeros(ts_batch_user_features.shape, dtype=bool) #Mask of seen items, e.g. False, True, False ...
    #         # ls_indizes_not_null = torch.nonzero(ts_batch_user_features, as_tuple=False)
    #         #
    #         # #TODO Too slow, find another way
    #         # for user_idx, item_idx in ls_indizes_not_null:
    #         #     mask_not_null[user_idx][item_idx] = True
    #
    #
    #
    #         ls_yhat_user = model.prediction_single(ts_batch_user_features)
    #         ls_yhat_user = ls_yhat_user * ts_batch_user_features  #Set all items to zero that are of no interest and haven't been seen
    #
    #         mse_user_wo_zeros += recmetrics.mse(ts_batch_user_features.tolist(),
    #                                       ls_yhat_user)  # Compute MSE on items that have been recommended
    #
    #
    #
    #     len_test_loader = len(test_loader)
    #     loss = loss / len_test_loader
    #     mse_user_w_zeros = mse_user_w_zeros /len_test_loader
    #     mse_user_wo_zeros = mse_user_wo_zeros /len_test_loader
    #     mar_k = mar_k / len_test_loader
    #     print("Test Reconstruction-Loss = {:.8f}".format(loss))
    #     print("MAR@{}: {}".format(k, mar_k))
    #     print("MSE-User with Zeros : {}".format(mse_user_w_zeros))
    #     print("MSE-User without Zeros (only evaluate items that have been seen) : {}".format(mse_user_wo_zeros))

def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar, unique_items).item()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


#%%
np_test_one = np.random.rand(1,2)
np_test_one
#%%

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/vae_1')
dct_hyperparam = {
"batch_size" : 512,
"epochs" : 20,
"learning_rate" : 1e-3,
"k": 500
}
train(dct_hyperparam, simplified_rating=False, small_dataset=True)
test()
#%%
# plot_ae_img(batch_features,test_loader)
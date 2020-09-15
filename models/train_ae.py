#pip install torch torchvision
import torch, torch.nn as nn, torchvision, torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import recmetrics
# import matplotlib.pyplot as plt
from surprise import Reader, SVD, Dataset
# from surprise.model_selection import train_test_split

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
unique_movies = 0
def create_user_item_matrix(df, simplified_rating: bool):
    # unique_movies = len(df["movieId"].unique())
    # global unique_items
    unique_items = df["movieId"].unique().max() + 1
    ls_users = df["userId"].unique()

    unique_users = len(df["userId"].unique()) +1
    np_user_item_mx = np.zeros((unique_users, unique_items), dtype=np.float32)#np.float_ is double

    print("Number of movies: {}".format(unique_items))
    print("Number of users: {}".format(unique_users))

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
    return np_user_item_mx


#%%
def plot_ae_img(batch_features, testloader):
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, 784)
            reconstruction = model(test_examples)
            break

    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(reconstruction[index].numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
#%%
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        #nn.Linear layer creates a linear function (θx + b), with its parameters initialized
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128 #input tensor with the size of [N, input_shape] where N is the number of examples, and input_shape is the number of features in one example.
        )

        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"] #out_features parameter dictates the feature size of the output tensor of a particular layer.
        )

    def forward(self, features):
        #Input
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        #Hidden - generate latent space
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        #Output - Take latent space
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        #Output
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

    #Input: All seen items of a user as vector
    def prediction_user(self, user_feature):
        user_preds = self.forward(user_feature)

        return user_preds

    #Input: One Hot Encoding
    def prediction_single(self, one_hot_movie_id):
        pred = self.forward(one_hot_movie_id)
        return pred

def get_default_hyperparam():
    return {
        "batch_size": 512,
        "epochs": 1,
        "learning_rate": 1e-3,
        "k": 10
    }

def generate_mnist_datasets():

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )
    return train_dataset, test_dataset

def generate_movielens_data(load_csv, small_dataset):
    pass

def generate_mask(ts_batch_user_features, tsls_yhat_user, user_based_items_filter: bool):
            # user_based_items_filter == True is what most people do
            mask = None
            if(user_based_items_filter):
                mask = ts_batch_user_features == 0. # filter out everything except what the user has seen , mask_zeros
            else:
                # TODO Mask filters also 1 out, that's bad
                mask = ts_batch_user_features == tsls_yhat_user #Obtain a mask for filtering out items that haven't been seen nor recommended, basically filter out what is 0:0 or 1:1
            return mask

#%%
train_dataset = None
test_dataset = None
def train(dct_hyperparam: dict, simplified_rating: bool, small_dataset: bool, load_csv: bool, use_mnist:bool, loss_user_items_only: bool):
    df_ratings = None
    # global train_dataset
    # global test_dataset
    #
    # global unique_items
    if(use_mnist):
        train_dataset, test_dataset = generate_mnist_datasets()
        unique_items = 784
    else:
        if(load_csv):
            if(small_dataset):
                print("Load small dataset")
                df_ratings = pd.read_csv("../data/movielens/small/ratings.csv")
            else:
                print("Load large dataset")
                df_ratings = pd.read_csv("../data/movielens/large/ratings.csv")

            np_user_item = create_user_item_matrix(df_ratings, simplified_rating=simplified_rating)
            # unique_items = np_user_item.shape[1]
            print("Save numpy matrix..")

            np.save("../data/numpy/small_ml_ratings.npy", np_user_item)
        else:
            print("Load user-item numpy matrix")
            np_user_item = np.load("../data/numpy/small_ml_ratings.npy")
            # unique_items = np_user_item.shape[1]

        unique_items = np_user_item.shape[1]

        train_dataset, test_dataset = train_test_split(np_user_item, test_size=0.33, random_state=42)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=1, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=1
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #  use gpu if available
    model = AE(input_shape=unique_items).to(device) # load it to the specified device, either gpu or cpu
    optimizer = optim.Adam(model.parameters(), lr= dct_hyperparam['learning_rate']) # create an optimizer object -Adam optimizer with learning rate 1e-3
    criterion = nn.MSELoss() # mean-squared error loss

    print("Training started...")
    #compute train loss
    for epoch in range(dct_hyperparam["epochs"]):
        loss = 0
        y_hat = []
        for ts_batch_user_features in train_loader:

            # reshape mini-batch data to [N, 784] matrix
            if(use_mnist):
                ts_batch_user_features = ts_batch_user_features[0]
                ts_batch_user_features = ts_batch_user_features.view(-1, 784).to(device)
                #TODO that's more a hack, actually it should be ts_batch_user_features, _ in the for loop
            else:
                # load it to the active device
                ts_batch_user_features = ts_batch_user_features.to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            #Originally:  outputs = model(batch_features), train_loss = criterion(outputs, batch_features), train_loss.backward()
            # model.double()
            t_outputs = model(ts_batch_user_features)
            y_hat.extend(t_outputs.tolist())
            # compute training reconstruction loss
            train_loss = criterion(t_outputs, ts_batch_user_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()


        # compute the epoch training loss
        loss = loss / len(train_loader)
    #TODO fix broken pipe line caused by recmetrics
        # display the epoch training loss
        print("\nEpoch : {}/{}, Reconstruction-Loss = {:.8f}".format(epoch + 1, dct_hyperparam["epochs"], loss))

        mse = recmetrics.mse(train_dataset.tolist(), y_hat)
        print("MSE Training Reconstruction: {}".format(mse))

    #Compute Test Loss
    mse_user_w_zeros = 0
    mse_user_wo_zeros = 0
    mar_k = 0
    k = dct_hyperparam["k"]
    with torch.no_grad():
        for ts_batch_user_features in test_loader: #testloader holds user-item matrix n,i, e.g. 202,193610 -> 32, 193610
            ts_batch_user_features = ts_batch_user_features.to(device)
            #Test reconstruction
            reconstruction = model(ts_batch_user_features) #compute recommendations
            test_loss = criterion(reconstruction, ts_batch_user_features)
            loss += test_loss.item()


            #Test user prediction
            tsls_yhat_user = model.prediction_user(ts_batch_user_features)
            #Compute MAR@K
            ls_idx_yhat = (-tsls_yhat_user).argsort() #argsort returns indices of the given list in ascending order. For descending we invert the list, so each element is inverted
            ls_idx_y = (-ts_batch_user_features).argsort()
            mar_k += recmetrics.mark(ls_idx_y.tolist(), ls_idx_yhat.tolist(), k)


            #Compute MSE
            mask = generate_mask(ts_batch_user_features, tsls_yhat_user, user_based_items_filter=loss_user_items_only)
            tsls_yhat_user_filtered = tsls_yhat_user[~mask] #Predicted: Filter out unseen+unrecommended items
            ts_user_features_seen = ts_batch_user_features[~mask] #Ground Truth: Filter out unseen+unrecommended items

            mse_user_w_zeros += recmetrics.mse(ts_user_features_seen.tolist(), tsls_yhat_user_filtered) #Compute MSE on items that have been recommended
            print("MSE Criterion w/ zeroes: {}".format(criterion(ts_batch_user_features, tsls_yhat_user)))
            print("MSE Criterion wo/ (filtered) zeroes: {}".format(criterion(ts_user_features_seen, tsls_yhat_user_filtered)))


            #Test single prediction

            #Snippet is to create a mask for filtering out items that haven't been seen
            #not necessary since we can just multiply the output by the input and have a natural filter
            # mask_not_null = np.zeros(ts_batch_user_features.shape, dtype=bool) #Mask of seen items, e.g. False, True, False ...
            # ls_indizes_not_null = torch.nonzero(ts_batch_user_features, as_tuple=False)
            #
            # #TODO Too slow, find another way
            # for user_idx, item_idx in ls_indizes_not_null:
            #     mask_not_null[user_idx][item_idx] = True



            ls_yhat_user = model.prediction_single(ts_batch_user_features)
            ls_yhat_user = ls_yhat_user * ts_batch_user_features  #Set all items to zero that are of no interest and haven't been seen

            mse_user_wo_zeros += recmetrics.mse(ts_batch_user_features.tolist(),
                                          ls_yhat_user)  # Compute MSE on items that have been recommended



        len_test_loader = len(test_loader)
        loss = loss / len_test_loader
        mse_user_w_zeros = mse_user_w_zeros /len_test_loader
        mse_user_wo_zeros = mse_user_wo_zeros /len_test_loader
        mar_k = mar_k / len_test_loader
        print("Test Reconstruction-Loss = {:.8f}".format(loss))
        print("MAR@{}: {}".format(k, mar_k))
        print("MSE-User with Zeros : {}".format(mse_user_w_zeros))
        print("MSE-User without Zeros (only evaluate items that have been seen) : {}".format(mse_user_wo_zeros))


#%%



#Be careful to run with a large k -> needs too long for MAP metric
train(get_default_hyperparam(), simplified_rating=False, small_dataset=True, load_csv=True, use_mnist=False, loss_user_items_only = True)
#%%
# plot_ae_img(batch_features,test_loader)
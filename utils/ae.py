#pip install torch torchvision
import torch, torch.nn as nn, torchvision, torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# import recmetrics
# import matplotlib.pyplot as plt
# from surprise import Reader, SVD, Dataset
# from surprise.model_selection import train_test_split


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
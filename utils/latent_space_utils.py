import matplotlib.pyplot as plt
# %matplotlib inline
# from utils import utils
# import utils.utils as utils
from lib.eval.hinton import hinton
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
import numpy as np
from lib.eval.regression import normalize, entropic_scores, print_table_pretty, nrmse
from lib.zero_shot import get_gap_ids
from lib.utils import mkdir_p
import math
import pandas as pd
from utils import plot_utils
import seaborn as sns
# split inputs and targets into sets: [train, dev, test, (zeroshot)]
def traverse(test_model, experiment_path, dct_param):
    ls_axes = []
    fig = plt.figure(figsize=(25, 10))
    # st = fig.suptitle('This is a somewhat long figure title', fontsize=16, y=1.3)
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
            z = [0 for i in range(0,test_model.no_latent_factors)]
            # z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
    ax_heatmap.get_figure().tight_layout()
    # plt.title('foo')
    # st.set_y(0.95)
    plt.subplots_adjust(top=0.5)

    plt.show()
    plot_utils.save_figure(ax_heatmap.get_figure(), experiment_path, 'traversing_heatmap', dct_param)


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

def alter_z(ts_z, latent_factor_position, model, strategy):
    tmp_z = ts_z.detach().clone()
    if(strategy == 'max'):
        print('--> Max strategy')
        ts_z[:, latent_factor_position] = model.z_max_train[latent_factor_position]/2 #32x no_latent_factors, so by accesing [:,pos] I get the latent factor for the batch of 32 users
    elif(strategy == 'min'):
        print('--> Min strategy')
        raise NotImplementedError("Min Strategy needs to be implemented")
    elif(strategy == 'min_max'):
        # print('--> Min- Max strategy')

        try:
            z_max_range = model.z_max_train[latent_factor_position]/2 #TODO Evtl. take z_max_train here
            z_min_range = model.z_min_train[latent_factor_position]/2

            if(np.abs(z_max_range) > np.abs(z_min_range)):
                ts_z[:, latent_factor_position] = model.z_max_train[latent_factor_position]/2
            else:
                ts_z[:, latent_factor_position] = model.z_min_train[latent_factor_position]/2
        except IndexError:
            print('stop')
    # print('Change in Z:{}'.format(tmp_z-ts_z))
    return ts_z
import matplotlib.pyplot as plt
# %matplotlib inline
# from utils import utils
# import utils.utils as utils
from lib.eval.hinton import hinton

import os
import numpy as np
from lib.eval.regression import normalize, entropic_scores, print_table_pretty, nrmse
from lib.zero_shot import get_gap_ids
from lib.utils import mkdir_p
import math
import pandas as pd

# split inputs and targets into sets: [train, dev, test, (zeroshot)]
def split_data(data, n_train, n_dev, n_test, zshot):
    train = data[:n_train]
    dev = data[n_train: n_train + n_dev]
    test = data[n_train + n_dev: n_train + n_dev + n_test]
    if zshot:
        pass
        # return [create_gap(train), create_gap(dev), create_gap(test), data[gap_ids]]
    return [train, dev, test, None]


# normalize input and target datasets [train, dev, test, (zeroshot)]
def normalize_datasets(datasets, zshot):
    datasets[0], mean, std, _ = normalize(datasets[0], remove_constant=False)
    datasets[1], _, _, _ = normalize(datasets[1], mean, std, remove_constant=False)
    datasets[2], _, _, _ = normalize(datasets[2], mean, std, remove_constant=False)
    if zshot:
        datasets[3], _, _, _ = normalize(datasets[3], mean, std, remove_constant=False)
    return datasets


def fit_visualise_quantify(regressor, params, err_fn, importances_attr, test_time=False, save_plot=False, n_models=1, n_z=0, m_codes=None, gts = None, zshot=False, model_names = None, n_c=None, figs_dir = None):
    # lists to store scores
    m_disent_scores = [] * n_models
    m_complete_scores = [] * n_models

    # arrays to store errors (+1 for avg)
    train_errs = np.zeros((n_models, n_z + 1))
    dev_errs = np.zeros((n_models, n_z + 1))
    test_errs = np.zeros((n_models, n_z + 1))
    zshot_errs = np.zeros((n_models, n_z + 1))

    # init plot (Hinton diag)
    fig, axs = plt.subplots(1, n_models, figsize=(12, 6), facecolor='w', edgecolor='k')
    # axs = axs.ravel()

    for i in range(n_models):
        # init inputs
        X_train, X_dev, X_test, X_zshot = m_codes[i][0], m_codes[i][1], m_codes[i][2], m_codes[i][3]

        # R_ij = relative importance of c_i in predicting z_j
        R = []

        for j in range(n_z):
            # init targets [shape=(n_samples, 1)]
            y_train = gts[0].iloc[:, j]
            y_dev = gts[1].iloc[:, j]
            y_test = gts[2].iloc[:, j] if test_time else None
            y_zshot = gts[3].iloc[:, j] if zshot else None

            # fit model
            model = regressor(**params[i][j])
            model.fit(X_train, y_train)

            # predict
            y_train_pred = model.predict(X_train)
            y_dev_pred = model.predict(X_dev)
            y_test_pred = model.predict(X_test) if test_time else None
            y_zshot_pred = model.predict(X_zshot) if zshot else None

            # calculate errors
            train_errs[i, j] = err_fn(y_train_pred, y_train)
            dev_errs[i, j] = err_fn(y_dev_pred, y_dev)
            test_errs[i, j] = err_fn(y_test_pred, y_test) if test_time else None
            zshot_errs[i, j] = err_fn(y_zshot_pred, y_zshot) if zshot else None

            # extract relative importance of each code variable in predicting z_j
            r = getattr(model, importances_attr)[:, None]  # [n_c, 1]
            R.append(np.abs(r))

        R = np.hstack(R)  # columnwise, predictions of each z

        # disentanglement
        disent_scores = entropic_scores(R.T)
        disent_scores = [0 if math.isnan(score) else score for score in disent_scores]
        c_rel_importance = np.sum(R, 1) / np.sum(R)  # relative importance of each code variable
        disent_w_avg = np.sum(np.array(disent_scores) * c_rel_importance)
        disent_scores.append(disent_w_avg)
        m_disent_scores.append(disent_scores)

        # completeness
        complete_scores = entropic_scores(R)
        complete_avg = np.mean(complete_scores)
        complete_scores.append(complete_avg)
        m_complete_scores.append(complete_scores)

        # informativeness (append averages)
        train_errs[i, -1] = np.mean(train_errs[i, :-1])
        dev_errs[i, -1] = np.mean(dev_errs[i, :-1])
        test_errs[i, -1] = np.mean(test_errs[i, :-1]) if test_time else None
        zshot_errs[i, -1] = np.mean(zshot_errs[i, :-1]) if zshot else None

        # visualise
        hinton(R, '$\mathbf{z}$', '$\mathbf{c}$', ax=axs, fontsize=18)
        axs.set_title('{0}'.format(model_names[i]), fontsize=20)

    plt.rc('text', usetex=True)
    if save_plot:
        fig.tight_layout()
        plt.savefig(os.path.join(figs_dir, "hint_{0}_{1}.pdf".format(regressor.__name__, n_c)))
    else:
        plt.show()

    print_table_pretty('Disentanglement', m_disent_scores, 'c', model_names)
    print_table_pretty('Completeness', m_complete_scores, 'z', model_names)

    print("Informativeness:")

    print_table_pretty('Training Error', train_errs, 'z', model_names)
    print_table_pretty('Validation Error', dev_errs, 'z', model_names)

    if test_time:
        print_table_pretty('Test Error', test_errs, 'z', model_names)
        if zshot:
            print_table_pretty('Zeroshot Error', zshot_errs, 'z', model_names)

def run_disentanglement_eval(test_model):

    print('fo')
    np_z_test = test_model.np_z_test
    test_y = test_model.test_y
    # setup
    seed = 123
    rng = np.random.RandomState(seed)
    data_dir = '../data/lasso'  # '../wgan/data/'
    codes_dir = os.path.join(data_dir, 'codes/')
    figs_dir = 'figs/'
    mkdir_p(figs_dir)

    n_c = 10
    zshot = False

    model_names = ['VAE']
    exp_names = [m.lower() for m in model_names]
    n_models = len(model_names)
    train_fract, dev_fract, test_fract = 0.8, 0.1, 0.1

    # load inputs (model codes)
    m_codes = []
    for n in exp_names:
        try:
            # m_codes.append(np.load(os.path.join(codes_dir, n + '.npy')))
            m_codes.append(np_z_test)
        except IOError:
            # .npz, e.g. pca with keys: codes, explained_variance
            m_codes.append(np.load(os.path.join(codes_dir, n + '.npz'))['codes'])

    # load targets (ground truths)
    # gts = np.load(os.path.join(data_dir, 'teapots.npz'))['gts']
    gts = test_y
    # n_samples = gts.shape[0]
    n_samples = len(test_y)
    # n_z = gts.shape[1]
    n_z = len(np.unique(test_y))
    n_train, n_dev, n_test = int(train_fract * n_samples), int(dev_fract * n_samples), int(test_fract * n_samples)

    # create 'gap' in data if zeroshot (unseen factor combinations)
    if zshot:
        try:
            gap_ids = np.load(os.path.join(data_dir, 'gap_ids.npy'))
        except IOError:
            gap_ids = get_gap_ids(gts)


        def create_gap(data):
            return np.delete(data, gap_ids, 0)

    gts = pd.get_dummies(pd.DataFrame(data=gts))
    gts = split_data(gts, n_train, n_dev, n_test, zshot)
    for i in range(n_models):
        m_codes[i] = split_data(m_codes[i], n_train, n_dev, n_test, zshot)

    # gts = normalize_datasets(gts, zshot)
    # for i in range(n_models):
    #     m_codes[i] = normalize_datasets(m_codes[i])


    ##Train Lasso
    from sklearn.linear_model import Lasso

    alpha = 0.02
    params = [[{"alpha": alpha}] * n_z] * n_models  # constant alpha for all models and targets
    importances_attr = 'coef_'  # weights
    err_fn = nrmse  # norm root mean sq. error
    test_time = True
    save_plot = False

    fit_visualise_quantify(Lasso,
                           params,
                           err_fn,
                           importances_attr,
                           test_time,
                           save_plot,
                           n_models,
                           n_z, m_codes,gts, zshot, model_names, n_c, figs_dir)
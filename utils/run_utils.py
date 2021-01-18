import argparse
import torch
import numpy as np
from torch import nn, optim
from utils import latent_space_utils, metric_utils, plot_utils, disentangle_utils, data_utils
import wandb
import time
import os

def create_training_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--max_epochs', type=int, default=100, metavar='N',
                        help='number of max epochs to train (default: 15)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=0, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    return args


def create_model_params(experiment_path, epoch, lf, beta, sigmoid_annealing_threshold, expanded_user_item, mixup,
                        no_generative_factors, max_epochs, is_hessian_penalty_activated, used_data=None, small_movie_dataset=True):
    model_params = {"simplified_rating": True,
                    "small_dataset": small_movie_dataset,
                    "test_size": 0.05,  # TODO Change test size to 0.33
                    "latent_dim": 3,
                    "beta": 1,
                    "sigmoid_annealing_threshold": 0,
                    "max_epochs": max_epochs}
    # model_params.update(args.__dict__)
    # print(**model_params)

    if(small_movie_dataset):
        model_params['test_size'] = 0.3
    else:
        model_params['test_size'] = 0.2
    # merged_params = (lambda first_dict, second_dict: {**first_dict, **second_dict})(args.__dict__,model_params)
    # print(merged_params)

    model_params['experiment_path'] = experiment_path
    model_params['max_epochs'] = epoch
    model_params['latent_dim'] = lf
    model_params['beta'] = beta
    model_params['synthetic_data'] = None
    model_params['sigmoid_annealing_threshold'] = sigmoid_annealing_threshold
    model_params['expanded_user_item'] = expanded_user_item
    model_params['mixup'] = mixup
    model_params['generative_factors'] = no_generative_factors
    model_params['is_hessian_penalty_activated'] = is_hessian_penalty_activated
    model_params['used_data'] = used_data



    return model_params

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

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.orthogonal_(m.weight)
        m.bias.data.zero_()

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


def load_model(VAE, model_path, attribute_path, experiment_path):
    print('------ Load model -------')
    vae = VAE.load_from_checkpoint(model_path)  # , load_saved_attributes=True, saved_attributes_path='attributes.pickle'
    # test_model.test_size = model_params['test_size']
    vae.load_attributes_and_files(attribute_path)
    vae.experiment_path_test = experiment_path
    return vae

def train_model(vae, trainer, model_path, attribute_path):
    # plot_utils.print_nn_summary(model, size =model.np_synthetic_data.shape[1])

    print('------ Start Training ------')
    trainer.fit(vae)
    kld_matrix = vae.KLD
    # print('% altering has provided information gain:{}'.format(int(settings.ig_m_hat_cnt) / (int(settings.ig_m_cnt) + int(settings.ig_m_hat_cnt))))

    print('------ Saving model ------')
    trainer.save_checkpoint(model_path)
    vae.save_attributes(attribute_path)

def test_model(test_model, trainer, wandb_logger, dct_param, experiment_path, synthetic_data):
    print('------ Start Test ------')

    start = time.time()
    # plot_utils.plot_samples(test_model, experiment_path, dct_param)

    if (test_model.no_latent_factors < 11):
        latent_space_utils.traverse(test_model, experiment_path, dct_param)
        if (test_model.np_synthetic_data is not None):
            metric_utils.create_multi_mce(test_model, dct_param)

    trainer.test(test_model)  # The test loop will not be used until you cadct_paramll.
    print('Test time in seconds: {}'.format(time.time() - start))
    # print('% altering has provided information gain:{}'.format( int(settings.ig_m_hat_cnt)/(int(settings.ig_m_cnt)+int(settings.ig_m_hat_cnt) )))
    # print(results)

    if (synthetic_data):
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

    dct_images = {img_path.split(sep='_')[2].split(sep='/')[-1]: wandb.Image(plt.imread(img_path)) for img_path in
                  ls_path_images}
    wandb.log(dct_images)
    #wandb.log({"example_1": wandb.Image(...), "example_2",: wandb.Image(...)})

    print('Test done')
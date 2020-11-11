import argparse

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
                        no_generative_factors, max_epochs, is_hessian_penalty_activated, used_data):
    model_params = {"simplified_rating": True,
                    "small_dataset": True,
                    "test_size": 0.2,  # TODO Change test size to 0.33
                    "latent_dim": 3,
                    "beta": 1,
                    "sigmoid_annealing_threshold": 0,
                    "max_epochs": max_epochs}
    # model_params.update(args.__dict__)
    # print(**model_params)

    # merged_params = (lambda first_dict, second_dict: {**first_dict, **second_dict})(args.__dict__,model_params)
    # print(merged_params)

    model_params['experiment_path'] = experiment_path
    model_params['max_epochs'] = epoch
    model_params['latent_dim'] = lf
    model_params['beta'] = beta
    # model_params['synthetic_data'] = None
    model_params['sigmoid_annealing_threshold'] = sigmoid_annealing_threshold
    model_params['expanded_user_item'] = expanded_user_item
    model_params['mixup'] = mixup
    model_params['generative_factors'] = no_generative_factors
    model_params['is_hessian_penalty_activated'] = is_hessian_penalty_activated
    model_params['used_data'] = used_data

    return model_params
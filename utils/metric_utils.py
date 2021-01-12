from __future__ import print_function
# from hessian_penalty_pytorch import hessian_penalty
# import recmetrics
# from surprise import Reader, SVD, Dataset
# from surprise.model_selection import train_test_split
from collections import defaultdict
import math
import utils.plot_utils as utils
import numpy as np
from scipy.stats import entropy
from utils import utils
from sklearn.metrics import mean_squared_error
from utils import settings, latent_space_utils, plot_utils
import torch
import pandas as pd

def mce_relative_frequency(y_hat, y_hat_latent, dct_attribute_distribution):
    # dct_dist = pickle.load(movies_distribution)

    dct_mce = defaultdict(float)
    for idx_vector in range(y_hat.shape[0]):
        for attribute in y_hat:
            if(attribute not in ['Unnamed: 0', 'unnamed_0', 'plot_outline']):
                ls_y_attribute_val = utils.my_eval(y_hat.iloc[idx_vector][attribute]) #e.g. Stars: ['Pitt', 'Damon', 'Jolie']
                ls_y_latent_attribute_val = utils.my_eval(y_hat_latent.iloc[idx_vector][attribute]) #e.g Stars: ['Depp', 'Jolie']
                mean = 0
                cnt_same = 0
                mce=0
                try:
                    #Two cases: Either cell contains multiple values, than it is a list
                    #or it contains a single but not in a list. In that case put it in a list
                    if(type(ls_y_latent_attribute_val) is not list):
                        ls_y_latent_attribute_val = [ls_y_latent_attribute_val]
                        ls_y_attribute_val =[ls_y_attribute_val]

                    #Go through elements of a cell
                    for value in ls_y_latent_attribute_val: #same as characteristic
                        if(value in ls_y_attribute_val): #if no change, assign highest error
                            # mean += 1
                            mce +=1
                            # ls_y_latent_attribute_val.pop(value)

                        else:
                            # characteristic = y_hat.loc[idx_vector, attribute]
                            relative_frequency = dct_attribute_distribution[attribute]['relative'][str(value)]
                            mce += relative_frequency
                            # print('\t Value: {}, Relative frequency:{}'.format(value, relative_frequency))
                    #if no values are presented in the current cell than assign highest error
                    if(len(ls_y_latent_attribute_val)==0):
                        mce =1
                    else:
                        mce = mce/len(ls_y_latent_attribute_val)
                    # print('Attribute: {}, mce:{}'.format(attribute, mce))

                    dct_mce[attribute] = mce
                except (KeyError, TypeError, ZeroDivisionError) as e:
                    print("Error Value:{}".format(value))

    return dct_mce

def shannon_inf_score(m, m_hat):
    epsilon = 1e-10
    shannon_inf = - math.log(m_hat) + epsilon
    mce = 1 / shannon_inf * math.exp(m_hat - m)

    if (mce < 0):
        print('fo')
    if (mce > 15):
        mce = 15
    if (math.isnan(mce)):
        print('nan detected')
    return mce

def calculate_normalized_entropy(population):
    H=0
    H_n =0
    try:
        if(len(population) == 1): #only one attribute is present
            H = - (population[0] * math.log(population[0], 2))
            return H
        else:
            # for rf in population:
            #     H_n = - (rf * math.log(rf, 2)) /math.log(len(population), 2)


            H_scipy = entropy(population, base=2)
            H_n = H_scipy / math.log(len(population), 2)
            # print(H_scipy, H_n)
            # assert H_scipy == H_n
    except ValueError:
        print('f')


    return H_n

def information_gain_single(m_sample, dct_population):
    ls_population_rf = [val for key, val in dct_population.items()]
    population_entropy = calculate_normalized_entropy(ls_population_rf)
    print('RF sample:{}'.format(m_sample))
    print('Population entropy: {}'.format(population_entropy))

    if(type(m_sample) is not list):
        m_sample = [m_sample]

    m_entropy = calculate_normalized_entropy(m_sample)
    print('Entropy sample:{}'.format(m_entropy))


    ig_m_sample = population_entropy - m_entropy
    print('IG sample:{} '.format(ig_m_sample))


    return ig_m_sample



def information_gain(m, m_hat, dct_population):
    ls_population_rf = [val for key, val in dct_population.items()]
    population_entropy = calculate_normalized_entropy(ls_population_rf)
    # print('RF original:{} RF new:{}'.format(m,m_hat))
    # print('Population entropy: {}'.format(population_entropy))

    if(type(m) is not list):
        m = [m]
    if(type(m_hat) is not list):
        m_hat = [m_hat]

    m_entropy = calculate_normalized_entropy(m)
    m_hat_entropy = calculate_normalized_entropy(m_hat)
    # print('Entropy original:{} new:{}'.format(m_entropy,m_hat_entropy))


    ig_m = population_entropy - m_entropy
    ig_m_hat = population_entropy - m_hat_entropy
    # print('IG original:{} new:{} '.format(ig_m, ig_m_hat))



    if(ig_m_hat > ig_m): #This means it was more unlikely so we gain information. Goal is to get to 1
        settings.ig_m_hat_cnt +=1
        # return ig_m_hat
    else:
        settings.ig_m_cnt += 1
    # print('IG Difference:{}\n{}\n'.format(ig_m_hat - ig_m, '-'*80))
    return  ig_m -ig_m_hat

def single_information_gain(y_hat_sampled, dct_attribute_distribution):
    # dct_dist = pickle.load(movies_distribution)

    dct_ig = defaultdict(float)
    for idx_vector in range(y_hat_sampled.shape[0]):
        for attribute in y_hat_sampled:
            print('Attribute:{}'.format(attribute))
            if(attribute not in ['Unnamed: 0', 'unnamed_0', 'plot_outline','id']):
                #Get attribute
                ls_y_attribute_val = utils.my_eval(y_hat_sampled.iloc[idx_vector][attribute]) #e.g. Stars: ['Pitt', 'Damon', 'Jolie']

                mean = 0
                cnt_same = 0
                mce=0
                m_hat = 0
                m = 0

                try:
                    #Two cases: Either cell contains multiple values, than it is a list
                    #or it contains a single but not in a list. In that case put it in a list
                    if(type(ls_y_attribute_val) is not list):
                        ls_y_attribute_val =[ls_y_attribute_val]

                    if (len(ls_y_attribute_val) == 0):
                        break

                    ls_m_hat_rf=[]
                    #Go through elements of a cell
                    for value in ls_y_attribute_val: #same as characteristic
                            y_hat_sample_attribute_relative_frequency = dct_attribute_distribution[attribute]['relative'][str(value)]
                            m_hat += y_hat_sample_attribute_relative_frequency
                            ls_m_hat_rf.append(y_hat_sample_attribute_relative_frequency)
                            # print('\t Value: {}, Relative frequency:{}'.format(value, relative_frequency))
                    #if no values are presented in the current cell than assign highest error
                    if(len(ls_y_attribute_val)==0):
                        # mce =15
                        #TODO sth else than just break, maybe mce = -1?
                        break
                    else:
                        #rf = relative frequency
                        dct_population = dct_attribute_distribution[attribute]['relative']

                        ls_y_hat_rf = [dct_population[str(val)] for val in ls_y_attribute_val]
                        m = np.asarray(ls_y_hat_rf).mean()
                        m_hat = m_hat/len(ls_y_attribute_val)

                        ig = information_gain_single(ls_y_hat_rf, dct_population)
                        # mce = shannon_inf_score(m, m_hat)

                    prev_ig = dct_ig.get(attribute)
                    if (prev_ig):
                        dct_ig[attribute] = (prev_ig + ig) / 2
                    else:
                        dct_ig[attribute] = ig
                except (KeyError, TypeError, ZeroDivisionError) as e:
                    print("Error Value:{}".format(value))

    return dct_ig


def mce_information_gain(y_hat, y_hat_latent, dct_attribute_distribution):
    # dct_dist = pickle.load(movies_distribution)

    dct_mce = defaultdict(float)
    for idx_vector in range(y_hat.shape[0]):
        for attribute in y_hat:
            # print('Attribute:{}'.format(attribute))
            if(attribute not in ['Unnamed: 0', 'unnamed_0', 'plot_outline','id']):
                #Get attribute
                ls_y_attribute_val = utils.my_eval(y_hat.iloc[idx_vector][attribute]) #e.g. Stars: ['Pitt', 'Damon', 'Jolie']
                ls_y_latent_attribute_val = utils.my_eval(y_hat_latent.iloc[idx_vector][attribute]) #e.g Stars: ['Depp', 'Jolie']
                # print('Attribute Values to compapre:{} - {}'.format(ls_y_attribute_val, ls_y_latent_attribute_val))
                mean = 0
                cnt_same = 0
                mce=0
                m_hat = 0
                m = 0

                try:
                    #Two cases: Either cell contains multiple values, than it is a list
                    #or it contains a single but not in a list. In that case put it in a list
                    if(type(ls_y_latent_attribute_val) is not list):
                        ls_y_latent_attribute_val = [ls_y_latent_attribute_val]
                        ls_y_attribute_val =[ls_y_attribute_val]

                    if (len(ls_y_attribute_val) == 0):
                        break

                    ls_m_hat_rf=[]
                    #Go through elements of a cell
                    for value in ls_y_latent_attribute_val: #same as characteristic
                        # if(value in ls_y_attribute_val): #if no change, assign highest error
                            # mean += 1
                            # m_hat +=1

                            # ls_y_latent_attribute_val.pop(value)

                        # else:
                            # characteristic = y_hat.loc[idx_vector, attribute]
                            y_hat_latent_attribute_relative_frequency = dct_attribute_distribution[attribute]['relative'][str(value)]
                            m_hat += y_hat_latent_attribute_relative_frequency
                            ls_m_hat_rf.append(y_hat_latent_attribute_relative_frequency)
                            # print('\t Value: {}, Relative frequency:{}'.format(value, relative_frequency))
                    #if no values are presented in the current cell than assign highest error
                    if(len(ls_y_latent_attribute_val)==0):
                        # mce =15
                        #TODO sth else than just break, maybe mce = -1?
                        break
                    else:
                        #rf = relative frequency
                        dct_population = dct_attribute_distribution[attribute]['relative']

                        ls_y_hat_rf = [dct_population[str(val)] for val in ls_y_attribute_val]
                        m = np.asarray(ls_y_hat_rf).mean()
                        m_hat = m_hat/len(ls_y_latent_attribute_val)

                        mce = information_gain(ls_y_hat_rf, ls_m_hat_rf, dct_population)
                        # mce = shannon_inf_score(m, m_hat)

                    prev_mce = dct_mce.get(attribute)
                    if (prev_mce):
                        dct_mce[attribute] = (prev_mce + mce) / 2
                    else:
                        dct_mce[attribute] = mce
                except (KeyError, TypeError, ZeroDivisionError) as e:
                    print("Error Value:{}".format(value))

    return dct_mce



def mce_shannon_inf(y_hat, y_hat_latent, dct_attribute_distribution):
    # dct_dist = pickle.load(movies_distribution)

    dct_mce = defaultdict(float)
    for idx_vector in range(y_hat.shape[0]):
        for attribute in y_hat:
            if(attribute not in ['Unnamed: 0', 'unnamed_0', 'plot_outline']):
                ls_y_attribute_val = utils.my_eval(y_hat.iloc[idx_vector][attribute]) #e.g. Stars: ['Pitt', 'Damon', 'Jolie']
                ls_y_latent_attribute_val = utils.my_eval(y_hat_latent.iloc[idx_vector][attribute]) #e.g Stars: ['Depp', 'Jolie']
                mean = 0
                cnt_same = 0
                mce=0
                m_hat = 0
                m = 0

                try:
                    #Two cases: Either cell contains multiple values, than it is a list
                    #or it contains a single but not in a list. In that case put it in a list
                    if(type(ls_y_latent_attribute_val) is not list):
                        ls_y_latent_attribute_val = [ls_y_latent_attribute_val]
                        ls_y_attribute_val =[ls_y_attribute_val]

                    if (len(ls_y_attribute_val) == 0):
                        break

                    #Go through elements of a cell
                    for value in ls_y_latent_attribute_val: #same as characteristic
                        if(value in ls_y_attribute_val): #if no change, assign highest error
                            # mean += 1
                            m_hat +=1
                            # ls_y_latent_attribute_val.pop(value)

                        else:
                            # characteristic = y_hat.loc[idx_vector, attribute]
                            y_hat_latent_attribute_relative_frequency = dct_attribute_distribution[attribute]['relative'][str(value)]
                            m_hat += y_hat_latent_attribute_relative_frequency
                            # print('\t Value: {}, Relative frequency:{}'.format(value, relative_frequency))
                    #if no values are presented in the current cell than assign highest error
                    if(len(ls_y_latent_attribute_val)==0):
                        mce =15
                    else:
                        #rf = relative frequency

                        ls_y_hat_rf = [dct_attribute_distribution[attribute]['relative'][str(val)] for val in ls_y_attribute_val]
                        m = np.asarray(ls_y_hat_rf).mean()
                        m_hat = m_hat/len(ls_y_latent_attribute_val)

                        mce = shannon_inf_score(m, m_hat)

                    dct_mce[attribute] = mce
                except (KeyError, TypeError, ZeroDivisionError) as e:
                    print("Error Value:{}".format(value))

    return dct_mce

def sampling_inf_gain(model):
    dct_mce_mean = defaultdict(lambda: defaultdict(lambda:dict))
    # hold n neurons of hidden layer
    # change 1 neuron

    z = model.z
    # dct_attribute_distribution = utils.load_json_as_dict('attribute_distribution.json') #    load relative frequency distributioon from dictionary (pickle it)
    print('Z: {}'.format(z))
    for latent_factor_position in range(model.no_latent_factors):
        print('Sampling strategy')

        for i in range(0, 2):
            ls_tensor_values= []
            for sample in range(0,200):
                z = [0 for i in range(0, model.no_latent_factors)]
                # z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ls_tensor_values.append(z)
            np_tensor_value = np.asarray(ls_tensor_values).astype(float)
            mu, sigma = 0, 0.2  # mean and standard deviation
            s = np.random.normal(mu, sigma, 200)

            if(i == 0):
                np_tensor_value[:,latent_factor_position] = -1 + s
            else:
                np_tensor_value[:,latent_factor_position] = 1 + s


            np_altered_z = np_tensor_value #np.asarray(ls_tensor_values)

            ts_altered_z = torch.tensor(np_altered_z).float()
            ls_y_hat_latent_changed = model.decode(ts_altered_z).detach().numpy()


            ls_idx_yhat_latent = (-ls_y_hat_latent_changed).argsort()  # argsort returns indices of the given list in ascending order. For descending we invert the list, so each element is inverted

            ls_dct_mce = []
            for user_idx in range(len(ls_y_hat_latent_changed)):  # tqdm(range(len(ls_idx_yhat)), total = len(ls_idx_yhat)):
                y_hat_latent = ls_y_hat_latent_changed[user_idx]
                y_hat_latent_k_highest = (-y_hat_latent).argsort()[:1]
                y_hat_latent_w_metadata = model.df_movies.loc[
                    model.df_movies['id'].isin(y_hat_latent_k_highest.tolist())]
                single_mce = single_information_gain(y_hat_latent_w_metadata,
                                                                  model.dct_attribute_distribution)  # mce for n columns


                ls_dct_mce.append(single_mce)


            orientation = 'negative' if i == 0 else 'positive'
            dct_mce_mean[latent_factor_position]
            # dct_mce_mean[latent_factor_position][orientation]
            dct_mce_mean[latent_factor_position][orientation] = dict(utils.calculate_mean_of_ls_dict(ls_dct_mce))
    return dct_mce_mean


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


#MCE is calculated for each category
def mce_batch(model, ts_batch_features, dct_index2itemId, k=0):
    dct_mce_mean = defaultdict()
    # hold n neurons of hidden layer
    # change 1 neuron
    ls_y_hat, mu, logvar, p, q = model(ts_batch_features)
    z = model.z
    # dct_attribute_distribution = utils.load_json_as_dict('attribute_distribution.json') #    load relative frequency distributioon from dictionary (pickle it)
    # print('Z: {}'.format(z))
    for latent_factor_position in range(model.no_latent_factors):

        # print("Calculate MCEs for latent factor: {}".format(latent_factor_position))
        ts_altered_z = latent_space_utils.alter_z(z.detach().clone(), latent_factor_position, model, strategy='min_max')#'max'
        # print('ts_altered_z: {}'.format(ts_altered_z))

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
            # single_mce = metric_utils.mce_shannon_inf(y_hat_w_metadata, y_hat_latent_w_metadata, model.dct_attribute_distribution) #mce for n columns
            # print('Calculate MCE for:\n{} \nand: \n{}'.format(y_hat_w_metadata, y_hat_latent_w_metadata))
            single_mce = mce_information_gain(y_hat_w_metadata, y_hat_latent_w_metadata, model.dct_attribute_distribution) #mce for n columns
            ls_dct_mce.append(single_mce)

            # print(single_mce)
        dct_mce_mean[latent_factor_position] = dict(utils.calculate_mean_of_ls_dict(ls_dct_mce))
    return dct_mce_mean


def create_multi_mce(test_model, dct_param):

    dct_inf_gain = sampling_inf_gain(test_model)
    dct_inf_gain = utils.default_to_regular(dct_inf_gain)
    ls_rows = []
    ls_columns = []

    for lf_key, orient_value in dct_inf_gain.items():

        for att_key, dct_val in orient_value.items():
            row = []
            col = []
            row.append(att_key)
            col.append('spectrum')
            for key, val in dct_val.items():
                row.append(val)
                col.append(key)

            row.append(lf_key)
            col.append('LF')
            ls_columns.append(col)

            ls_rows.append(row)

    df_ig = pd.DataFrame(data=ls_rows, columns=ls_columns[0])
    plot_utils.plot_ig_by_latent_factor(df_ig, 'Title', test_model.experiment_path_test, dct_param)
    # df_mce_results = pd.read_json(
    #     experiment_path + '/mce_results.json')  # ../data/generated/mce_results.json'


def calculate_metrics(y_actual, y_predicted):
    #RMSE
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)
    # print("RMSE :{}".format(rmse))

    #MSE
    mse = mean_squared_error(y_actual, y_predicted, squared=True)
    # print("MSE :{}".format(mse))
    return rmse,mse

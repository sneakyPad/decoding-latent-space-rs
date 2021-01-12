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
from utils import settings


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

def calculate_metrics(y_actual, y_predicted):
    #RMSE
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)
    # print("RMSE :{}".format(rmse))

    #MSE
    mse = mean_squared_error(y_actual, y_predicted, squared=True)
    # print("MSE :{}".format(mse))
    return rmse,mse

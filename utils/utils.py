import torch
import json
import seaborn as sns
import pandas as pd
from sklearn import decomposition
from sklearn.metrics import mean_squared_error
import math
from torchsummaryX import summary
from datetime import datetime
import random
from tqdm import tqdm
import ast
import itertools
from collections import Counter
import plotly.express as px
import matplotlib.pyplot as plt
import os
import numpy as np


def my_eval(expression):
    try:
        return ast.literal_eval(str(expression))
    except SyntaxError: #e.g. a ":" or "(", which is interpreted by eval as command
            return expression
    except ValueError: #e.g. an entry is nan, in that case just return an empty string
        return ''


def compute_relative_frequency(df_meta):
    print('Compute relative frequency for all columns and their attributes...')
    #Goal is:
    #Cast:
        # Tom Hanks: 0,3%
        # Matt Damon: 0,2%
    # fpp =np.vstack(df_meta['genres'].values)
    # np_array = df_meta['genres'].values
    # tmp_list = []
    # for element in np_array:
    #     tmp_list.extend(eval(element))

    # print(fpp)len_crawled_ids
    #TODO Implement my eval: https://stackoverflow.com/questions/31423864/check-if-string-can-be-evaluated-with-eval-in-python
    dct_rel_freq={}
    for column in tqdm(df_meta.columns, total=len(df_meta.columns)):
        print('Column: {}'.format(column))
    #     ls_ls_casted=[]
    #     for str_elem in df_meta[column].values:
    #         str_elem= str(str_elem)#.replace(':','').replace('(','').replace(')','')
    #         try:
    #             ls_ls_casted.append(ast.literal_eval(str_elem))
    #         except SyntaxError:
    #             ls_ls_casted.append(str_elem)

        ls_ls_casted = [my_eval(str_elem) for str_elem in df_meta[column].values] #cast encoded lists to real list
        # ls_ls_casted = [json.loads(str(str_elem)) for str_elem in df_meta[column].values] #cast encoded lists to real list
        try:
            if(type(ls_ls_casted[0]) == list):
                merged_res = itertools.chain(*ls_ls_casted) #join all lists to one single list
                ls_merged = list(merged_res)
            else:
                ls_merged = ls_ls_casted
            if(column not in ['Unnamed: 0', 'unnamed_0']):
                c = Counter(ls_merged)
                dct_counter = {str(key): value for key, value in c.items()}
                dct_rel_freq[column]={}
                dct_rel_freq[column]['absolute'] = dct_counter
                # print('Column: {}\n\t absolute:{}'.format(dct_rel_freq[column]['absolute']))

                dct_rel_attribute = {str(key): value / sum(c.values()) for key, value in dct_counter.items()} #TODO create a dict with key val
                dct_rel_freq[column]['relative'] = dct_rel_attribute
                # print('\t relative:{}'.format(dct_rel_freq[column]['relative']))

        except TypeError:
            print('TypeError for Column:{} and ls_ls_casted:{} and *ls_ls_casted:{}'.format(column, ls_ls_casted, *ls_ls_casted))


    return dct_rel_freq
    # save_dict_as_json(dct_rel_freq, 'relative_frequency.json')

def create_experiment_directory():
    # datetime object containing current date and time
    now = datetime.now()

    print("now =", now)
    dt_string = now.strftime("%d-%m-%Y-%H_%M_%S")
    print("date and time =", dt_string)

    # define the name of the directory to be created
    path = "results/generated/" + dt_string + "/"

    try:
        os.mkdir(path)
        os.mkdir(path+"images/")

    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    return path



def load_json_as_dict(name):
    with open('../data/generated/' + name, 'r') as file:
        id2names = json.load(file)
        return id2names

def save_dict_as_json(dct, name, path=None):
    if(path):
        path = path
    else:
        path = '../data/generated/'

    with open(path + name, 'w') as file:
        json.dump(dct, file, indent=4, sort_keys=True)
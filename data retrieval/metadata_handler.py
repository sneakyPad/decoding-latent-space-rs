# pip install git+https://github.com/alberanid/imdbpy
# pip install imdbpy
from imdb import IMDb, IMDbDataAccessError
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import ast
from collections import defaultdict
import multiprocessing
dct_no_entries = defaultdict(int)
import numpy as np
import itertools
import os
from collections import Counter
from time import sleep
import math
import json
# manager = multiprocessing.Manager()
# shared_list = manager.list()
def benchmark_string_comparison():
    import time
    dct_foo = {'alpha':3,
               'beta':1,
               'gamma':2}
    a = id('Tomin Hanks')
    b= id('Tomin Cruise')
    avg_a = avg_b= avg_c = avg_d =avg_e= avg_f=0
    for i in range(0,10000):
        start = time.time()
        com = dct_foo['alpha']==dct_foo['beta']
        avg_a +=start - time.time()


        start = time.time()
        com = 3==1
        avg_b += start - time.time()

        start = time.time()
        com = 'alpha' == 'beta'
        avg_c += start - time.time()

        start = time.time()
        com = 'Tomin d. Hanks' == 'Tomin d. Cruise'
        avg_d += start - time.time()

        start = time.time()
        com = id('Tomin Hanks') == id('Tomin Cruise')
        avg_e += start - time.time()

        start = time.time()
        com = a == b
        avg_f += start - time.time()

    print(i)
    print(id('foo'))
    avg_a = (avg_a / i) *1000
    avg_b = (avg_b/i) * 1000
    avg_c = (avg_c/i) * 1000
    avg_d = (avg_d/i) * 1000
    avg_e = (avg_e / i) * 1000
    print(' Avg_a:{} \n Avg_b:{} \n Avg_c:{} \n Avg_d:{} \n Avg_e:{} \n Avg_f:{}'.format(avg_a,avg_b,avg_c,avg_d, avg_e, avg_f ))
# benchmark_string_comparison()
#%%
    import pandas as pd

    # df = df_meta.value_counts()
    # print(df.head())
def save_dict_as_json(dct, name):
    with open('../data/movielens/small/' + name, 'w') as file:

        json.dump(dct, file, indent=4, sort_keys=True)
def load_json_as_dict(name):
    with open('../data/movielens/small/' + name, 'r') as file:
        id2names = json.loads(file)
        return id2names
def load_dataset(small_dataset):
    if (small_dataset):
        print("Load small dataset")
        #%%
        df_movies = pd.read_csv("../data/movielens/small/links.csv")
    else:
        print("Load large dataset")
        df_movies = pd.read_csv("../data/movielens/large/links.csv")

    return df_movies
def fetch_example():
    # create an instance of the IMDb class
    ia = IMDb()

    # get a movie
    movie = ia.get_movie('0133093')
    print(ia.get_movie_keywords('0133093'))



    print('movie \n{}'.format(movie))
    # print the names of the directors of the movie
    print('Directors:')
    for director in movie['directors']:
        print(director['name'])

    # print the genres of the movie
    print('Genres:')
    for genre in movie['genres']:
        print(genre)

    # search for a person name
    people = ia.search_person('Mel Gibson')
    for person in people:
       print(person.personID, person['name'])


def beautify_names(dct_data, key):
    # clean actors:
    # start_time = time.time()
    ls_names = []
    try:
        for actor in dct_data[key]:
            if(bool(actor)):
                ls_names.append(actor['name'])
    except KeyError:
        dct_no_entries[key]+=1
        # print()No entries for key:

    # print("--- %s seconds ---" % (time.time() - start_time))
    # total_time_one +=time.time() - start_time
    return ls_names

def remove_keys(dict, keys):
    if(keys == None):
        keys = ['certificates', 'cover url', 'thanks',
                      'special effects companies', 'transportation department',
                      'make up department', 'special effects', 'stunts', 'costume departmen',
                      'location management', 'editorial department', 'casting directors', 'art directors',
                      'production managers', 'art department', 'sound department',
                      'visual effects', 'camera department', 'costume designers'
                      'casting department', 'miscellaneous', 'akas', 'production companies', 'distributors',
                      'other companies', 'synopsis', 'cinematographers', 'production designers',
                      'custom designers', 'Opening Weekend United Kingdom', 'Opening Weekend United States']
    for key in keys:
        dict.pop(key, None)
    return dict

def fetch_by_imdb_ids(ls_ids):
    imdb = IMDb()
    ls_metadata =[]

    import random
    # cnt_connection_reset=0
    for id in tqdm(ls_ids, total = len(ls_ids)):     # loop through ls_ids
        try:
            sleep_t = random.randint(0,10)/10

            # sleep(sleep_t)  # Time in seconds
            #TODO Actually it should be checked whether this is single process or not, bc the IMDB Peer error occurs only w/ multiprocessing
            movie = imdb.get_movie(id)

            # TODO Optional: select metadata

            dct_data = movie.data

            # to be cleaned:
            keys_to_beautify = ['cast','directors', 'writers', 'producers', 'composers', 'editors',
                                'animation department', 'casting department','music department', 'set decorators',
                                'script department', 'assistant directors', 'writer', 'director', 'costume designers']
            for key in keys_to_beautify:
                dct_data[key] = beautify_names(dct_data, key)

            #unwrap box office:
            try:
                dct_data.update(dct_data['box office'])
                del dct_data['box office']
            except KeyError:
                pass
                # print('Unwrap: key error for movieId:{} '.format(movie.movieID))# dct_data['title']

            dct_data = remove_keys(dct_data, None)

            #Fetch stars of the movie with bs4
            ls_stars = fetch_stars(id)
            dct_data['stars'] =ls_stars
            dct_data['imdbId'] = id

            #add dict to the list of all metadata
            ls_metadata.append(dct_data)
        except Exception:
            print('Exception for id:{}'.format(id))
            # cnt_connection_reset+=1
    return ls_metadata, dct_no_entries


#extracts baed on column_name a nested list of the attribute, e.g. cast and creates
# a second list with the respective ids that are looked up in actor2id.
# you can extract more columns by adding them to column_name
def ids2names(df_movies, actor2id, column_name):
    dct_ls_ids = defaultdict(list)
    dct_ls_columns = defaultdict(list)
    for idx, row in tqdm(df_movies.iterrows(), total=df_movies.shape[0]):
        for column in column_name:  # column_name: ['cast','stars']
            if (type(row[column]) == list):
                ls_names = row[column]
            else:
                ls_names = ast.literal_eval(
                    row[column])  # literal_eval casts the list which is encoded as a string to a list

            # ls_names = row[column]
            dct_ls_columns[column] = ls_names
            # dct_ls_columns[column]= dct_ls_columns[column].append(ls_names)

        # if(type(row['cast'])==list):
        #     casts = row['cast']
        # else:
        #     casts = ast.literal_eval(row['cast']) #literal_eval casts the list which is encoded as a string to a list
        # if(type(row['stars'])==list):
        #     stars = row['stars']
        # else:
        #     stars = ast.literal_eval(row['stars'])

        for key, ls_names in dct_ls_columns.items():
            dct_ls_ids[key].append([actor2id[name] for name in dct_ls_columns[key]])
        # ls_ls_cast_ids.append([actor2id[name] for name in casts])
        # ls_ls_stars_ids.append([actor2id[name] for name in stars])

    return dct_ls_columns, dct_ls_ids

def names2ids(df, column_name):
    print('--- Transform names to ids and add an extra column for it ---')
    # df_movies = pd.read_csv("../data/movielens/small/df_movies.csv")
    df_movies = df
    actor2id = defaultdict(lambda: 1+len(actor2id))

    # [ls_casts[0].append(ls) for ls in ls_casts]
    ls_names = []

    #Add all names to one single list
    print('... Collect names:')
    for idx, row in tqdm(df_movies.iterrows(), total=df_movies.shape[0]):
        for column in column_name:
            if(type(row[column])==list):
                ls_names.extend(row[column])
            else:
                ls_names.extend(ast.literal_eval(row[column])) #literal_eval casts the list which is encoded as a string to a list

    # ls_elem = ls_elem.replace("[",'').replace("'",'').split(sep=',')
    c = Counter(ls_names)
    dct_bar = dict(c)
    for elem in list(ls_names):
        actor2id[elem] #Smart because, lambda has everytime a new element was added, a new default value
        # actor2id[elem] = actor2id[elem] + 1 #count the occurence of an actor
        # if (actor2id[elem] == 0): #assign an unique id to an actor/name
        #     actor2id[elem] = len(actor2id)

    print(actor2id)
    id2actor = {value: key for key, value in actor2id.items()}
    save_dict_as_json(actor2id, 'names2ids.json')
    save_dict_as_json(id2actor, 'ids2names.json')
    # print(id2actor[2])

    print("... Assign Ids to names:")
    dct_ls_columns, dct_ls_ids = ids2names(df_movies,actor2id,column_name)

    # lists look like this:
    # dct_ls_columns = {'cast':['wesley snipes','brad pitt'...]
    # dct_ls_ids ={'cast':[22,33,...]}
    for key, ls_names in dct_ls_columns.items():
        df_movies[key+"_id"] = dct_ls_ids[key]

    return df_movies

def fetch_stars(id):

    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }
    ls_stars = []

    try:
        url = "https://www.imdb.com/title/tt0{}/?ref_=rvi_tt".format(id)
        req = requests.get(url, headers)
        soup = BeautifulSoup(req.content, 'html.parser')
        h4_stars = soup.find("h4", text='Stars:')
        div_tag = h4_stars.parent
        next_a_tag = div_tag.findNext('a')
        while (next_a_tag.name != 'span'):
            if (next_a_tag.name == 'a'):
                ls_stars.append(str(next_a_tag.contents[0]))#str() casts from NavigabelString to string
            next_a_tag = next_a_tag.next_sibling
            # class 'bs4.element.Tag'>
        # next_a_tag.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling['class'][0] == 'ghost'
        # print(ls_stars)
    except AttributeError:
        print('AttributeError, movieId:{}'.format(id))
    finally:
        return ls_stars

# TODO Unfinished: This is not done yet
def enhance_by_stars(df):
    pass
    # tqdm.pandas(desc="my bar!")
    # df['stars'] = df['movieID'].apply(lambda id: fetch_stars(id))
    # return df

def print_exception_statistic(dct_no_entries):
    print('[--------- Exception Statistics ---------]')
    # print('No. of ConnectionResetError: {}'.format(cnt_reset))
    print('Joined No-Keys-Exception for the following keys:')
    for key, value in dct_no_entries.items():
        print("\tKey: {}, count:{}, relative: {}".format(key, value, value / len(ls_imdb_ids)))
    print('[----------------------------------------]')


def worker(ids):
    # global shared_list https://stackoverflow.com/questions/40630428/share-a-list-between-different-processes-in-python
    metadata, dct_no_entries = fetch_by_imdb_ids(ids)
    # shared_list.extend(metadata)
    # print('worker done')
    return metadata, dct_no_entries

def crawl_metadata(ls_imdb_ids, multi_processing, no_processes, develop_size):

    print('Fetching metadata of {} movies'.format(len(ls_imdb_ids)))
    if(develop_size>0):
        ls_imdb_ids = ls_imdb_ids[:develop_size]

    if (multi_processing):
        print('Start multiprocessing...')
        start_time = time.time()  # measure time

        no_processes = no_processes
        ls_ls_metadata = []
        ls_dct_exceptions = []
        cnt_reset = 0

        len_dataset = len(ls_imdb_ids)
        ls_splitted = np.array_split(np.array(ls_imdb_ids), no_processes)

        pool = multiprocessing.Pool(processes=no_processes)
        # m = multiprocessing.Manager()
        # q = m.Queue()

        # Pool.map returns list of pairs: https://stackoverflow.com/questions/39303117/valueerror-too-many-values-to-unpack-multiprocessing-pool
        for ls_metadata, dct_no_entries in pool.map(worker,ls_splitted):  # ls_ls_metadata=pool.map(worker, ls_splitted):
            # append both objects to a separate list
            ls_ls_metadata.append(ls_metadata)
            ls_dct_exceptions.append(dct_no_entries)


        print("--- %s seconds ---" % (time.time() - start_time))

        merged_res = itertools.chain(*ls_ls_metadata)  # unpack the list to merge n lists
        ls_metadata = list(merged_res)

        df_exceptions = pd.DataFrame(ls_dct_exceptions).sum()  # sum over all rows

        print_exception_statistic(df_exceptions.to_dict())
        print("--- %s seconds ---" % (time.time() - start_time))

    else:
        start_time = time.time()

        ls_metadata, dct_no_entries = fetch_by_imdb_ids(ls_imdb_ids)
        print_exception_statistic(dct_no_entries)

        print("--- %s seconds ---" % (time.time() - start_time))

    df_meta = pd.DataFrame(ls_metadata)
    print('Shape of crawled dataset:{}'.format(df_meta.shape[0]))
    return df_meta

def my_eval(expression):
    try:
        return ast.literal_eval(str(expression))
    except SyntaxError: #e.g. a ":" or "(", which is interpreted by eval as command
            return expression
    except ValueError: #e.g. an entry is nan, in that case just return an empty string
        return ''

def compute_relative_frequency(df_meta):
    #Goal is:
    #Cast:
        # Tom Hanks: 0,3%
        # Matt Damon: 0,2%
    # fpp =np.vstack(df_meta['genres'].values)
    # np_array = df_meta['genres'].values
    # tmp_list = []
    # for element in np_array:
    #     tmp_list.extend(eval(element))

    # print(fpp)
    #TODO Implement my eval: https://stackoverflow.com/questions/31423864/check-if-string-can-be-evaluated-with-eval-in-python
    dct_rel_freq={}
    for column in df_meta.columns:
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
            c = Counter(ls_merged)
            dct_counter = {str(key): value for key, value in c.items()}
            dct_rel_freq[column]={}
            dct_rel_freq[column]['absolute'] = dct_counter

            dct_rel_attribute = {str(key): value / sum(c.values()) for key, value in dct_counter.items()} #TODO create a dict with key val
            dct_rel_freq[column]['relative'] = dct_rel_attribute
        except TypeError:
            print('TypeError for Column:{} and ls_ls_casted:{} and *ls_ls_casted:{}'.format(column, ls_ls_casted, *ls_ls_casted))



    save_dict_as_json(dct_rel_freq, 'relative_frequency.json')
        # tmp_list = []
        # for element in df_meta[column]:
        #
        #     ls_ls_casted = [eval(str_elem) for str_elem in df_meta[element].values]
        #     itertools.chain(*ls_ls_casted)
        #     if(type(element)==str):
        #         tmp_list.extend(eval(element))
        #    # tmp_list.value

        # dct_rel_freq[element] =
        # df_meta[column] = tmp_list

    # df = df_meta['cast'][0].value_counts()
    # print(df)



if __name__ == '__main__':
    # benchmark_string_comparison()

    df_meta = pd.read_csv('../data/movielens/small/df_movies.csv')
    # compute_relative_frequency(df_meta)
    print('<----------- Metadata Crawler has started ----------->')
    # df_meta = None

    # fetch_example()
    #Load Dataset
    small_dataset = True
    multi_processing = True
    develop_size = 5000
    metadata = None
    crawl = True
    no_processes = 32
    df_links = load_dataset(small_dataset=small_dataset)

    if(crawl):
        #Enhance existing dataset by fetching metadata
        ls_imdb_ids = list(df_links['imdbId'])
        df_meta = crawl_metadata(ls_imdb_ids,
                                 multi_processing=multi_processing,
                                 no_processes=no_processes,
                                 develop_size=develop_size)
        print('Fetching Metadata done.')

    # enhance_by_stars(df_meta)

    #transform names to ids

    df_meta = names2ids(df=df_meta, column_name=['cast','stars'])
    dct_attribute_distribution = compute_relative_frequency(df_meta)
    # Save data
    print('Save enhanced movielens dataset...')
    if (small_dataset):
        df_meta.to_csv("../data/movielens/small/df_movies.csv")
        save_dict_as_json(dct_attribute_distribution, 'attribute_distribution.json')
    else:
        df_meta.to_csv("../data/movielens/large/df_movies.csv")
    print('<----------- Processing finished ----------->')

    #%%

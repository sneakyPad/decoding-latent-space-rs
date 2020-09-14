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

import json
# manager = multiprocessing.Manager()
# shared_list = manager.list()

def save_dict_as_json(dct, name):
    with open('../data/openlens/small/' + name, 'w') as file:

        json.dump(dct, file, indent=4, sort_keys=True)


def load_json_as_dict(name):
    with open('../data/openlens/small/' + name, 'r') as file:
        id2names = json.loads(file)
        return id2names


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

    # import sys
    # import pyprind
    # bar = pyprind.ProgBar(df_movies.shape[0], stream=sys.stdout)
    # for i in range(n):
    #     time.sleep(0.1)  # do some computation
    #     bar.update()
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
                                'animation department', 'music department', 'set decorators',
                                'script department', 'assistant directors', 'writer', 'director']
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


def load_dataset(small_dataset):
    if (small_dataset):
        print("Load small dataset")
        #%%
        df_movies = pd.read_csv("../data/openlens/small/links.csv")
    else:
        print("Load large dataset")
        df_movies = pd.read_csv("../data/openlens/large/links.csv")

    return df_movies


def names2ids(df, column_name):
    print('--- Transform names to ids and add an extra column for it ---')
    # df_movies = pd.read_csv("../data/openlens/small/df_movies.csv")
    df_movies = df

    # {print(element) for element in df_sub_cast}
    # print(pd.get_dummies(df_movies[:1]['cast']))
    # print(df_movies[0]['cast'].unique())

    actor2id = defaultdict(lambda: 1+len(actor2id))

    # [ls_casts[0].append(ls) for ls in ls_casts]
    str_test =""
    ls_names = []

    #Add all names to one single list
    print('Collect names:')
    for idx, row in tqdm(df_movies.iterrows(), total=df_movies.shape[0]):
        for column in column_name:
            if(type(row[column])==list):
                ls_names.extend(row[column])
            else:
                ls_names.extend(ast.literal_eval(row[column])) #literal_eval casts the list which is encoded as a string to a list

    # for idx, row in tqdm(df_movies.iterrows(), total = df_movies.shape[0]):
    #     ls_exp.extend(ast.literal_eval(row['stars']))

    # fpp = ','.join(df_movies['cast'].replace("[",''))

    # for ls_elem in df_sub_cast:
    # pd.unique(df_movies['cast'].values[0])

    # ls_elem = ast.literal_eval(casts)
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
    print(id2actor[2])

    print("Assign Ids to names:")
    ls_ls_cast_ids=[]
    ls_ls_stars_ids=[]
    dct_ls_ids = defaultdict(list)
    dct_ls_columns=defaultdict(list)
    for idx, row in tqdm(df_movies.iterrows(), total=df_movies.shape[0]):
        for column in column_name: #column_name: ['cast','stars']
            if (type(row[column]) == list):
                ls_names = row[column]
            else:
                ls_names = ast.literal_eval(row[column]) #literal_eval casts the list which is encoded as a string to a list

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

    for key, ls_names in dct_ls_columns.items():
        df_movies[key+"_id"] = dct_ls_ids[key]
    # df_movies['stars_id'] = ls_ls_stars_ids

    print('fo')
    return df_movies

        # ls_names.extend(ast.literal_eval(row['stars']))


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

def enhance_by_stars(df):
    # TODO This is not done yet
    tqdm.pandas(desc="my bar!")
    df['stars'] = df['movieID'].apply(lambda id: fetch_stars(id))
    return df

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

if __name__ == '__main__':
    print('<----------- Metadata Crawler has started ----------->')
    # df_meta = None

    # fetch_example()
    #Load Dataset
    small_dataset = True
    multi_processing = True
    develop_size = 3
    metadata = None
    crawl = True
    no_processes = 3
    df_movies = load_dataset(small_dataset=small_dataset)

    if(crawl):
        #Enhance existing dataset by fetching metadata
        ls_imdb_ids = list(df_movies['imdbId'])
        df_meta = crawl_metadata(ls_imdb_ids,
                                 multi_processing=multi_processing,
                                 no_processes=no_processes,
                                 develop_size=develop_size)
        print('Fetching Metadata done.')

    # enhance_by_stars(df_meta)

    #transform names to ids

    df_meta = names2ids(df=df_meta, column_name=['cast','stars'])

    # Save dataframe
    print('Save enhanced openlens dataset...')
    if (small_dataset):
        df_meta.to_csv("../data/openlens/small/df_movies.csv")
    else:
        df_meta.to_csv("../data/openlens/large/df_movies.csv")
    print('<----------- Processing finished ----------->')

    #%%
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

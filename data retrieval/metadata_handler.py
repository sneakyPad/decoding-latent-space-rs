# pip install git+https://github.com/alberanid/imdbpy
# pip install imdbpy
from imdb import IMDb
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

# manager = multiprocessing.Manager()
# shared_list = manager.list()
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

    for id in tqdm(ls_ids, total = len(ls_ids)):     # loop through ls_ids

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
            print('Unwrap: key error for movieId:{} '.format(movie.movieID))# dct_data['title']

        dct_data = remove_keys(dct_data, None)

        #Fetch stars of the movie with bs4
        ls_stars = fetch_stars(id)
        dct_data['stars'] =ls_stars

        #add dict to the list of all metadata
        ls_metadata.append(dct_data)


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


def ls_strings_2_ids(ls_strings: str = 'fo'):
    # %%
    import pandas as pd
    import tqdm

    df_movies = pd.read_csv("../data/openlens/small/df_movies.csv")
    df_sub_cast = df_movies[:1]['cast']

    # {print(element) for element in df_sub_cast}
    # print(pd.get_dummies(df_movies[:1]['cast']))
    # print(df_movies[0]['cast'].unique())

    id2actor = {}
    actor2id = defaultdict(int)
    ls_casts = df_movies['cast'].values
    # [ls_casts[0].append(ls) for ls in ls_casts]
    str_test =""
    ls_exp = []




    for idx, row in tqdm(df_movies.iterrows(), total = df_movies.shape[0]):
        ls_exp.extend(ast.literal_eval(row['stars']))
    # fpp = ','.join(df_movies['cast'].replace("[",''))

    # for ls_elem in df_sub_cast:
    # pd.unique(df_movies['cast'].values[0])

    # ls_elem = ast.literal_eval(casts)
    # ls_elem = ls_elem.replace("[",'').replace("'",'').split(sep=',')
    from collections import Counter
    c = Counter(ls_exp)
    dct_bar = dict(c)
    for elem in list(ls_exp):
        actor2id[elem] = actor2id[elem] + 1
        if (actor2id[elem] == 0):
            actor2id[elem] = len(actor2id)

    print(actor2id)

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
    print('Joined No-Keys-Exception for the following keys:')
    for key, value in dct_no_entries.items():
        print("Key: {}, count:{}, relative: {}".format(key, value, value / len(ls_imdb_ids)))


def worker(ids):
    # global shared_list https://stackoverflow.com/questions/40630428/share-a-list-between-different-processes-in-python
    metadata, dct_no_entries = fetch_by_imdb_ids(ids)
    # shared_list.extend(metadata)
    print('worker done')
    return metadata, dct_no_entries

def crawl_metadata(ls_imdb_ids, multiprocessing, test):

    print('Fetching metadata of {} movies'.format(len(ls_imdb_ids)))
    if(test):
        ls_imdb_ids = ls_imdb_ids[:50]

    if (multi_processing):
        start_time = time.time()  # measure time

        no_processes = 16
        ls_ls_metadata = []
        ls_dct_exceptions = []

        len_dataset = len(ls_imdb_ids)
        ls_splitted = np.array_split(np.array(ls_imdb_ids), no_processes)

        pool = multiprocessing.Pool(processes=no_processes)
        # m = multiprocessing.Manager()
        # q = m.Queue()

        # Pool.map returns list of pairs: https://stackoverflow.com/questions/39303117/valueerror-too-many-values-to-unpack-multiprocessing-pool
        for ls_metadata, dct_no_entries in pool.map(worker,
                                                    ls_splitted):  # ls_ls_metadata=pool.map(worker, ls_splitted):
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
    # df_meta = None

    # fetch_example()
    #Load Dataset
    small_dataset = True
    multi_processing = True
    metadata = None
    df_movies = load_dataset(small_dataset=small_dataset)

    #Enhance existing dataset by fetching metadata
    ls_imdb_ids = list(df_movies['imdbId'])
    df_meta = crawl_metadata(ls_imdb_ids, multiprocessing=multiprocessing, test=False)

    #Save dataframe
    if(small_dataset):
        df_meta.to_csv("../data/openlens/small/df_movies.csv")
    else:
        df_meta.to_csv("../data/openlens/large/df_movies.csv")
    # enhance_by_stars(df_meta)


    print('Fetching Metadata done.')
    # ls_strings_2_ids()

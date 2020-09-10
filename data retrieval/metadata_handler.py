# pip install git+https://github.com/alberanid/imdbpy
# pip install imdbpy
from imdb import IMDb
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup


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
        print('No entries for key: {}'.format(key))

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
    for id in ls_ids:     # loop through ls_ids

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
            print('key error for movieId:{} '.format(movie.movieID))# dct_data['title']

        dct_data = remove_keys(dct_data, None)
        ls_metadata.append(dct_data)

    return ls_metadata


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

    df_movies = pd.read_csv("../data/openlens/small/df_movies.csv")
    df_sub_cast = df_movies[:1]['cast']

    # {print(element) for element in df_sub_cast}
    # print(pd.get_dummies(df_movies[:1]['cast']))
    # print(df_movies[0]['cast'].unique())
    import ast
    from collections import defaultdict
    id2actor = {}
    actor2id = defaultdict(int)
    ls_casts = df_movies['cast'].values
    # [ls_casts[0].append(ls) for ls in ls_casts]
    str_test =""
    ls_exp = []
    for idx, row in df_movies.iterrows():
        ls_exp.extend(ast.literal_eval(row['cast']))
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
    print('f')
def fetch_stars(id):
    #TODO id
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }
    url = "https://www.imdb.com/title/tt0133093/?ref_=rvi_tt"
    req = requests.get(url, headers)
    soup = BeautifulSoup(req.content, 'html.parser')
    fo = soup.find("h4", text='Stars:')
    div_tag = fo.parent
    next_a_tag = div_tag.findNext('a')
    ls_stars = []
    while (next_a_tag.name != 'span'):

        if (next_a_tag.name == 'a'):
            ls_stars.append(next_a_tag.contents[0])
        next_a_tag = next_a_tag.next_sibling
        # class 'bs4.element.Tag'>
    # next_a_tag.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling['class'][0] == 'ghost'
    print(ls_stars)
    # fo = soup.find('inline').get_text()
    print(fo)

if __name__ == '__main__':




    # fetch_example()
    # #Load Dataset
    # small_dataset = True
    # df_movies = load_dataset(small_dataset=small_dataset)
    #
    # #Enhance existing dataset by fetching metadata
    # ls_imdb_ids = list(df_movies['imdbId'])
    # print('Fetching metadata of {} movies'.format(len(ls_imdb_ids)))
    # metadata = fetch_by_imdb_ids(ls_imdb_ids[:1])
    # df_meta = pd.DataFrame(metadata)
    #
    # #Save dataframe
    # if(small_dataset):
    #     df_meta.to_csv("../data/openlens/small/df_movies.csv")
    # else:
    #     df_meta.to_csv("../data/openlens/large/df_movies.csv")
    #
    # print('Fetching Metadata done.')
    # ls_strings_2_ids()
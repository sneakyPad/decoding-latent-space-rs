# pip install git+https://github.com/alberanid/imdbpy
# pip install imdbpy
from imdb import IMDb
import pandas as pd
import time

def fetch_example():
    # create an instance of the IMDb class
    ia = IMDb()

    # get a movie
    movie = ia.get_movie('0133093')
    ia.get_movie_keywords('0133093')

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


#clean actors:
# start_time = time.time()
def beautify_names(dct_data, key):
    ls_names = []

    try:
        for actor in dct_data[key]:
            if(bool(actor)):
                ls_names.append(actor['name'])
    except KeyError:
        print('No entries for key: {}'.format(key))

    return ls_names

 # print("--- %s seconds ---" % (time.time() - start_time))
        # total_time_one +=time.time() - start_time
def remove_keys(dict, keys):
    if(keys == None):
        keys = ['certificates', 'cover url', 'thanks',
                      'special effects companies', 'transportation department',
                      'make up department', 'special effects', 'stunts', 'costume department',
                      'location management', 'editorial department', 'casting directors', 'art directors',
                      'production managers', 'art department', 'sound department',
                      'visual effects', 'camera department',
                      'casting department', 'miscellaneous', 'akas', 'production companies', 'distributors',
                      'other companies', 'synopsis', 'cinematographers', 'production designers',
                      'custom designers']
    for key in keys:
        dict.pop(key, None)
    return dict



def fetch_by_imdb_ids(ls_ids):
    imdb = IMDb()
    ls_metadata =[]
    for id in ls_ids:     # loop through ls_ids

        movie = imdb.get_movie(id)

        # Optional: select metadata

        # store metadata as entry and a flatted list
        dct_data = movie.data
        dct_data = remove_keys(dct_data, None)




        # to be cleaned:
        keys_to_beautify = ['cast','directors', 'writers', 'producers', 'composers', 'editors',
                            'animation department', 'music department', 'set decorators',
                            'script department', 'assistant directors', 'writer', 'director']
        for key in keys_to_beautify:
            dct_data[key] = beautify_names(dct_data, key)




        # casts = dct_data['cast']
        # for i in range(0, len(casts)):
        #     ls_casts.append(casts[i].__str__())
        # dct_data['cast'] = ls_casts



        #unwrap box office:
        try:
            dct_data.update(dct_data['box office'])
            del dct_data['box office']
            ls_metadata.append(dct_data)
        except KeyError:
            print('key error for movieId:{} with title:{} '.format(movie.movieID, dct_data['title']))



    return ls_metadata


def load_dataset(small_dataset):
    if (small_dataset):
        print("Load small dataset")
        df_movies = pd.read_csv("../data/openlens/small/links.csv")
    else:
        print("Load large dataset")
        df_movies = pd.read_csv("../data/openlens/large/links.csv")

    return df_movies

if __name__ == '__main__':
    df_movies = load_dataset(small_dataset=True)
    ls_imdb_ids = list(df_movies['imdbId'])
    metadata = fetch_by_imdb_ids(ls_imdb_ids[:20])
    df_meta = pd.DataFrame(metadata)
    print('fo')
    # add metadata to df
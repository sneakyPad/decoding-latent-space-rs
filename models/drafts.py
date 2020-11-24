from collections import defaultdict
from utils import plot_utils
def mean(*lst):
    columns = lst.key
    return sum(lst) / len(lst)
# if __name__ == '__main__':
#
#     ls_dct=[{'Stars':2, 'Cast':0.11},
#          {'Stars':3, 'Cast':0.01},
#          {'Stars':5, 'Cast':0.01}
#             ]
#
#     # result =map(mean, **ls_dct)
#     # print(list(result))
#     dct_sum = defaultdict(float)
#
#     import numpy as np
#     test = np.array(ls_dct).mean()
#
#     for dict in ls_dct:
#         for key, val in dict.items():
#             dct_sum[key] += val
#     np_mean_vals = np.array(list(dct_sum.values()))/len(ls_dct)
#     dct_mean = list(zip(dct_sum.keys(), np_mean_vals))
#     print(dct_mean)
#

import random
# import random
def create_synthetic_data():
    no_samples =50
    genres = ['Crime', 'Mystery', 'Thriller', 'Action', 'Drama', 'Romance','Comedy', 'War','Adventure', 'Family']
    year = ['1980', '1990', '2000', '2010', '2020']
    stars = ['Tom Hanks', 'Tim Allen', 'Don Rickles','Robin Williams', 'Kirsten Dunst', 'Bonnie Hunt']
    rating = ['7', '8', '9', '10']

    dct_base_data ={'genres': genres, 'year': year, 'stars': stars, 'rating': rating}
    ls_movies = []

    #genre-users
    ls_attributes = ['genres', 'year', 'stars', 'rating']
    n_users = 300
    n_movies = len(ls_attributes * no_samples)
    np_user_item = np.zeros((n_users,n_movies))


    for attribute in ls_attributes:
        for i in range(no_samples):
            movie = {}
            movie[attribute] = [dct_base_data[attribute][0]]

            for other_attribute in ls_attributes:
                if(other_attribute == attribute):
                    continue
                if(other_attribute == 'rating' or other_attribute == 'year'):
                    movie[other_attribute] = random.choices(dct_base_data[other_attribute], k=1)
                else:
                    movie[other_attribute] = random.choices(dct_base_data[other_attribute], k=2)


            ls_movies.append(movie)


    df_synthentic_data = pd.DataFrame(columns=['genres', 'year', 'stars', 'rating'], data=ls_movies)
    df_synthentic_data['id'] = df_synthentic_data.index

    no_users_attribute_specific = int(n_users / len(ls_attributes))
    for i in range(0, len(ls_attributes)):
        end = (i+1) * no_samples
        start = end - no_samples
        sr_ids = df_synthentic_data.loc[start:end]['id']

        for idx in range(no_users_attribute_specific):
            no_of_seen_items = int(random.uniform(20, 30))
            seen = random.sample(list(sr_ids.values), k=no_of_seen_items)
            user_idx = i * no_users_attribute_specific + idx
            np_user_item[user_idx,seen] = 1

    # print(ls_movies)

    return np_user_item

import utils.plot_utils as utils

def plot_distribution(np_d, name):
    df_distribution = pd.DataFrame(data=np_d.T, columns=['LF:1', 'LF:2'])
    df_distribution['label'] = ""
    df_distribution.iloc[:int(no_samples / 2)]['label'] = '1'
    df_distribution.iloc[int(no_samples / 2):]['label'] = '0'

    g= sns.pairplot(df_distribution, hue='label', corner=False)  # kind='kde'
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    g._legend.remove()
    plt.tight_layout()
    # plt.axis('equal')
    # plt.axis('square')
    plot_utils.save_figure(g, 'results/example-imgs/', name, None)
    plt.show()

def visualize_distribution():

    label_one_x =[-4.6733332700e-1,
                        -1.3005096260e+0,
                        -3.0050890430e-1,
                        -6.5857001820e-1,
                         6.0554612610e-1]
    label_one_y = [2.0651949230e+0,
                        -1.1696251940e+0,
                         1.1064605260e-1,
                         5.4754081320e-1,
                        -1.5641052020e-1]

    label_two_x =[-1.0175966500e+0,
    -4.0908302830e-1,
    -4.9347227720e-1,
     1.8408036850e+0,
    -3.0763158310e-1
    ]

    label_two_y = [ 7.5227611050e-1,
     8.2668338170e-1,
     2.6345638890e+0,
     1.5311090330e+0,
     7.5191696740e-1]

    # np_d = np.array([label_one_x+label_two_x,label_one_y+label_two_y])

    #Cases: auseinander, aufeinander, perfekt

    no_samples = 300
    #aufeinander
    x = np.random.normal(0, 1, no_samples)
    y = np.random.normal(0, 1, no_samples)
    np_d = np.array([x,y])

    plot_distribution(np_d, 'overregularized')

    #auseinander
    x = np.random.normal(0, 0.3, no_samples)
    y = np.random.normal(0, 0.3, no_samples)
    x[:int(no_samples/2)] = x[:int(no_samples/2)]-3
    y[:int(no_samples/2)] = y[:int(no_samples/2)]+3
    # y = np.random.normal(-2, 1, no_samples)
    np_d = np.array([x, y])

    plot_distribution(np_d, 'not-regularized')


    #perfekt
    x = np.random.normal(0, 1, no_samples)
    y = np.random.normal(0, 1, no_samples)

    x[:int(no_samples / 2)] = x[:int(no_samples / 2)] +3

    x[int(no_samples / 2):] = x[int(no_samples / 2):]

    np_d = np.array([x, y])

    plot_distribution(np_d, 'regularized')

    print('f')
    #
if __name__ == '__main__':
    #%%

    import numpy as np
    from utils.morphomnist import io, morpho, perturb
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    # penguins = sns.load_dataset("penguins")
    #
    # visualize_distribution()

    perturbations = (
        lambda m: m.binary_image,  # No perturbation
        perturb.Thinning(amount=.7),
        perturb.Thickening(amount=1.),
        perturb.Swelling(strength=3, radius=7),
        perturb.Fracture(num_frac=3)
    )
    base_path = "/Users/d069735/workspace/Study/decoding-latent-space-rs/data/morpho-mnist/global/"
    images = io.load_idx(base_path+"train-images-idx3-ubyte.gz")
    perturbed_images = np.empty_like(images)
    perturbation_labels = np.random.randint(len(perturbations), size=len(images))
    for n in range(10000):
        morphology = morpho.ImageMorphology(images[n], scale=4)
        perturbation = perturbations[perturbation_labels[n]]
        perturbed_hires_image = perturbation(morphology)
        perturbed_images[n] = morphology.downscale(perturbed_hires_image)
    io.save_idx(perturbed_images, "output_dir/pm-images-idx3-ubyte.gz")
    io.save_idx(perturbation_labels, "output_dir/pm-pert-idx1-ubyte.gz")


    import pandas as pd
    import numpy as np

    from sklearn.datasets import load_iris
    import seaborn as sns

    create_synthetic_data()
    # iris = load_iris()
    # iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
    #                     columns=iris['feature_names'] + ['target'])
    # print(iris.columns)
    # # recast into long format
    # df = iris.melt(['target'], var_name='cols', value_name='vals')
    #
    # df.head()

    from sklearn import manifold, decomposition

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd

    import math
    from scipy.stats import entropy


    # def test_mce_populations():
    ls_pop0 = [0.5, 0.49, 0.01]
    ls_p = [0.5, 0.499, 0.001]
    ent = entropy(ls_pop0, base =2)

    ent_0 = entropy([ls_pop0[0], ls_pop0[1]], base =2 )/ math.log(len(ls_pop0), 2)

    ig_0 = ent - ent_0

    ent_1 = entropy([ls_pop0[2]], base =2 )
    ent_11 = - ls_pop0[2] * math.log(ls_pop0[2], 2)
    ent_111 = entropy([ls_pop0[2], 1- ls_pop0[2]], base=2) / math.log(len(ls_pop0), 2)
    assert ent_11 == ent_1
    print('ent_0:{} ent_1:{} '.format(ent_0,ent_1))
    ig_1 = ent - ent_1
    print("Entropy:{} \n, IG_1:{}, IG_2:{}".format(ent,ig_0, ig_1))





    ls_pop1 = [0.9, 0.1]
    ls_pop2 = [0.7, 0.3]
    ls_pop3 = [0.5, 0.1, 0.1,0.2, 0.1]
    ls_pop4 = [0.001, 0.1, 0.1,0.6, 0.199]
    ls_pop = [ls_pop0, ls_pop1, ls_pop2, ls_pop3, ls_pop4]

    dict_ent = {}
    dict_ent_n = {}
    for idx, population in enumerate(ls_pop):
        H = 0
        H_n =0
        for rf in population:
            H += rf * math.log(rf,2)
            H_n += rf * math.log(rf,2)/math.log(len(population),2)
            #http://www.endmemo.com/bio/shannonentropy.php

        H=-H
        H_n = -H_n
        dict_ent[idx] = H
        dict_ent_n[idx] = H_n
        print('H:{}'.format(H))
        print('H_n:{}'.format(H_n))
        print('-' * 40)

    print('-'*80)

    #own strategy
    mean = np.asarray(ls_pop1).mean()
    ent_own = (1/dict_ent[0])* -math.log(ls_pop1[1]) - math.log(ls_pop1[0])
    ent_own_n = (dict_ent_n[0])* -math.log(ls_pop1[1]) - math.log(ls_pop1[0])
    mean_own = (1/mean )* -math.log(ls_pop1[1]) - math.log(ls_pop1[0])
    print('Ent',ent_own)
    print('Ent norm:', ent_own_n)
    print('Mean',mean_own)
    print('-'*40)

    mean = np.asarray(ls_pop2).mean()
    ent_own = (1 / dict_ent[1]) * -math.log(ls_pop2[1]) - math.log(ls_pop2[0])
    ent_own_n = (dict_ent[1]) * -math.log(ls_pop2[1]) - math.log(ls_pop2[0])
    mean_own = (1 / mean) * -math.log(ls_pop2[1]) - math.log(ls_pop2[0])
    print('Ent', ent_own)
    print('Ent norm:', ent_own_n)

    print('Mean', mean_own)
    print('-'*40)



    mean = np.asarray(ls_pop3).mean()
    ent_own = (1 / dict_ent[2]) * -math.log(ls_pop3[2]) - math.log(ls_pop3[3])
    ent_own_n = (dict_ent[2]) * -math.log(ls_pop3[2]) - math.log(ls_pop3[3])
    mean_own = (1 / mean) * -math.log(ls_pop3[2]) - math.log(ls_pop3[3])
    print('Ent', ent_own)
    print('Ent norm:', ent_own_n)

    print('Mean', mean_own)
    print('-'*40)


    mean = np.asarray(ls_pop4).mean()
    ent_own = (1 / dict_ent[3]) * -math.log(ls_pop4[1]) - math.log(ls_pop4[0])
    ent_own_n = (dict_ent[3]) * -math.log(ls_pop4[1]) - math.log(ls_pop4[0])
    mean_own = (1 / mean) * -math.log(ls_pop4[1]) - math.log(ls_pop4[0])
    print('Ent', ent_own)
    print('Ent norm:', ent_own_n)

    print('Mean', mean_own)
    print('-'*40)


    print('Turn around:')
    ent_own = (1/dict_ent[0])* -math.log(ls_pop1[0]) - math.log(ls_pop1[1])
    mean_own = (1/mean )* -math.log(ls_pop1[0]) - math.log(ls_pop1[1])
    print('Ent',ent_own)
    print('Mean',mean_own)

    mean = np.asarray(ls_pop2).mean()
    ent_own = (1 / dict_ent[1]) * -math.log(ls_pop2[0]) - math.log(ls_pop2[1])
    mean_own = (1 / mean) * -math.log(ls_pop2[0]) - math.log(ls_pop2[1])
    print('Ent', ent_own)
    print('Mean', mean_own)

    mean = np.asarray(ls_pop3).mean()
    ent_own = (1 / dict_ent[2]) * -math.log(ls_pop3[3]) - math.log(ls_pop3[2])
    mean_own = (1 / mean) * -math.log(ls_pop3[3]) - math.log(ls_pop3[2])
    print('Ent', ent_own)
    print('Mean', mean_own)

    mean = np.asarray(ls_pop4).mean()
    ent_own = (1 / dict_ent[3]) * -math.log(ls_pop4[0]) - math.log(ls_pop4[1])
    mean_own = (1 / mean) * -math.log(ls_pop4[0]) - math.log(ls_pop4[1])
    print('Ent', ent_own)
    print('Mean', mean_own)

    #entropy

    #linear




    f =1

#______________________________#______________________________#______________________________

    foo = np.array([[1, 20, -20], [3, 4, -6], [3, 4, -9]])

    import numpy as np
    pca = decomposition.PCA(n_components=2)
    # print(np_x)
    pca.fit(foo)
    X = pca.transform(foo)
    # print(X)
    # sns.set_context("poster")
    sns.set_style("whitegrid")
    df_pca = pd.DataFrame(X, columns=['pca_1', 'pca_2'])
    sns.scatterplot(data=df_pca, x="pca_1", y="pca_2")
    # plt.scatter(X[:, 0], X[:, 1])
    plt.show()


    #TSNE
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    X_embedded = manifold.TSNE(n_components=2, random_state=42).fit_transform(X)
    df_tsne = pd.DataFrame(X_embedded, columns=["tsne_1", "tsne_2"])
    print(X_embedded)
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    # plt.show()


    #
    # z_mu = vae.get_z_mean(np.rint(foo).astype(np.float32))
    # tsne = manifold.TSNE(n_components=2, random_state=42)
    # z_tsne = tsne.fit_transform(z_mu)




#%%


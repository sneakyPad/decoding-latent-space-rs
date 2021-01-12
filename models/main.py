from models import train_movies_ae
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook" #set default renderer as notebook

import numpy as np
import pandas as pd

def plot_results():
    sns.set_theme(style='whitegrid')
    plt.figure()
    fig, ax = plt.subplots(figsize=(25, 12))

    df_results = pd.read_csv('../data/csv_results/results.csv', delimiter=';')
    df_results = df_results.iloc[::-1]
    df_results.plot.barh(y=['Avg. Disentanglement', 'Avg. Completeness'])
    plt.xlim(0, 1)
    plt.legend(bbox_to_anchor=(0.5, -0.15),  # 1.05, 1
               loc='upper center',  # 'center left'
               borderaxespad=0.,
               fontsize=9,
               ncol=2)
    plt.ylabel('Experiment')
    plt.xlabel('Value')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.show()


def main():
    # train_movies_ae.train(train_movies_ae.get_default_hyperparam(),
    #                       simplified_rating=False,
    #                       small_dataset=True,
    #                       load_csv=False,
    #                       use_mnist=False,
    #                       loss_user_items_only = True)

    plot_results()

if __name__ == "__main__":
    # execute only if run as a script
    main()